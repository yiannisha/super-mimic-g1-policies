"""Offline policy handoff/state/socket tests. Never initializes DDS or motors."""
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import socket
import sys
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from supermimic.deploy.core import Clip, DeployConfig, DeployCore, SafetyCfg, static_clip
from supermimic.deploy.live_switch import PolicyHandoff

BRIDGE = Path(__file__).resolve().parents[1] / 'scripts/deploy/bridge'
sys.path.insert(0, str(BRIDGE))
sys.path.insert(0, str(BRIDGE.parent))
from policy_control import ControlSocket
from g1_runner_bridge import Runner


@pytest.fixture
def cfg():
    return DeployConfig([f'joint{i}' for i in range(29)], np.zeros(29), np.ones(29),
                        np.ones(29), np.ones(29)*.1, np.ones(29)*10,
                        np.ones(29)*-5, np.ones(29)*5, np.ones(29)*10)


def clip(name, pose=0.):
    goal = np.zeros((3, 68), np.float32)
    goal[:, :29] = pose
    goal[:, 60] = -1
    goal[:, 67] = .8
    return Clip(name, goal, np.full((3,29), pose), np.tile([0.,0.,-1.], (3,1)))


def policy(value):
    def call(obs, goal):
        return np.full(29, value)
    return call


def state():
    return SimpleNamespace(q=np.zeros(29), dq=np.zeros(29), gyro=np.zeros(3),
                           quat=np.array([1.,0.,0.,0.]), temp=np.full((29,2),30.))


def step(core):
    s = state()
    if isinstance(core, PolicyHandoff):
        return core.step(s.q, s.dq, s.gyro, s.quat)
    return core.step(s.q, s.dq, s.gyro, s.quat, advance=False)


def runner(cfg):
    r = Runner.__new__(Runner)
    r.cfg=cfg
    r.a=SimpleNamespace(no_policy=False,policy_switch_s=.2,sim_inject_nan_step=None,
                        ignore_temperature_safety=False)
    r.lock=threading.RLock()
    r.state='STANDBY_POST'; r.state_t=time.monotonic()-20
    r.clips=[clip('idle'),clip('walk',.2)]
    r.static_pre=[static_clip(c,0) for c in r.clips]
    r.clip_idx=0
    r.bank=None; r.bank_aliases=[]; r.control=None
    r.switching=None; r.switch_target_idx=None; r.thermal=None
    r.policy_fn=policy(.1); r.n_policy_calls=0
    r.policy=r.counted_policy(r.policy_fn)
    r.core=DeployCore(cfg,r.policy,static_clip(r.clips[0],2))
    step(r.core)
    r.safety=SafetyCfg()
    r.sb_win=deque([(0,0,0)]); r.temp_hist=deque([(0,np.zeros(29))])
    r.sbs=object(); r.confirm=object()
    r.standby_allowed=lambda *args,**kw:(True,'steady')
    r.events=[]
    r.event=lambda kind,msg:r.events.append((kind,msg))
    return r


def test_handoff_preserves_history_and_both_action_endpoints(cfg):
    old=DeployCore(cfg,policy(.1),clip('a'))
    step(old)
    before=old.hist.copy()
    new=DeployCore(cfg,policy(.3),clip('b',.2))
    handoff=PolicyHandoff(old,new,.2)
    np.testing.assert_array_equal(new.hist,before)
    previous=old.last_action.copy()
    actions=[]
    while not handoff.done:
        result=step(handoff)
        assert result.abort is None
        np.testing.assert_allclose(new.hist[-1,-29:],previous)
        np.testing.assert_allclose(old.last_action,result.action)
        np.testing.assert_allclose(new.last_action,result.action)
        actions.append(result.action.copy())
        previous=result.action.copy()
    np.testing.assert_allclose(actions[0],.1)
    np.testing.assert_allclose(actions[-1],.3)
    assert np.all(np.diff(np.array(actions),axis=0)>=-1e-12)
    np.testing.assert_allclose(new.clip.joint_pos[0],.2)
    assert new.frame==0
    with pytest.raises(RuntimeError,match='completed'):
        step(handoff)


def test_same_policy_handoff_only_runs_one_inference_per_tick(cfg):
    calls=[]
    def fn(obs,goal):
        calls.append(1)
        return np.full(29,.1)
    old=DeployCore(cfg,fn,clip('a')); step(old)
    new=DeployCore(cfg,fn,clip('b',.1))
    handoff=PolicyHandoff(old,new,.2,same_policy=True)
    initial=len(calls)
    while not handoff.done:
        step(handoff)
    assert len(calls)-initial==handoff.steps
    np.testing.assert_array_equal(new.hist,old.hist)


def test_bad_candidate_action_keeps_abort(cfg):
    old=DeployCore(cfg,policy(.1),clip('a')); step(old)
    new=DeployCore(cfg,policy(np.nan),clip('b'))
    handoff=PolicyHandoff(old,new,.2)
    assert 'non-finite action' in step(handoff).abort


def test_incompatible_cfg_is_rejected_before_mutation(cfg):
    old=DeployCore(cfg,policy(0),clip('a'))
    original=old.clip
    new=DeployCore(replace(cfg,kp=cfg.kp*2),policy(0),clip('b'))
    with pytest.raises(ValueError,match='kp differs'):
        PolicyHandoff(old,new,.2)
    assert old.clip is original


def test_opposite_reference_gravity_is_rejected(cfg):
    target=clip('upside-down'); target.proj_grav[:]=[0,0,1]
    with pytest.raises(ValueError,match='tilt limit'):
        PolicyHandoff(DeployCore(cfg,policy(0),clip('a')),DeployCore(cfg,policy(0),target),.2)


@pytest.mark.parametrize('status',['PASSIVE','INTERP','HOLD','STANDBY_PRE','POLICY','END_HOLD','DAMP'])
def test_selection_is_not_queued_in_any_other_state(cfg,status):
    r=runner(cfg); r.state=status
    original=r.core
    ok,why=r.begin_policy_switch(1,state(),time.monotonic())
    assert not ok and 'only allowed in STANDBY_POST' in why
    assert r.switching is None and r.core is original and r.clip_idx==0


@pytest.mark.parametrize('repeat',[False,True])
def test_selection_and_same_motion_replay_return_to_pre_without_playback(cfg,repeat):
    r=runner(cfg)
    index=0 if repeat else 1
    assert r.begin_policy_switch(index,state(),time.monotonic())[0]
    assert r.state=='STANDBY_POST'
    assert not r.begin_policy_switch(index,state(),time.monotonic())[0]
    while not r.switching.done:
        step(r.switching)
    assert r.finish_policy_switch()
    assert r.state=='STANDBY_PRE' and r.clip_idx==index and r.core.frame==0
    assert r.switching is None
    assert not r.sb_win and not r.temp_hist and r.confirm is None


def test_thermal_stop_and_failed_stability_gate_leave_old_policy_running(cfg):
    r=runner(cfg); old=r.core
    r.thermal={'motor':0}
    assert not r.begin_policy_switch(1,state(),time.monotonic())[0]
    assert r.core is old and r.switching is None
    r.thermal=None
    r.standby_allowed=lambda *a,**kw:(False,'still moving')
    assert r.begin_policy_switch(1,state(),time.monotonic())==(False,'still moving')
    assert r.core is old and r.switching is None


def test_damp_during_handoff_cannot_commit_new_policy(cfg):
    r=runner(cfg); old=r.core
    assert r.begin_policy_switch(1,state(),time.monotonic())[0]
    while not r.switching.done:
        step(r.switching)
    r.state='DAMP'
    assert not r.finish_policy_switch()
    assert r.core is old and r.clip_idx==0 and r.state=='DAMP'


def test_new_post_window_discards_pre_motion_samples(cfg):
    r=runner(cfg); r.state='POLICY'
    assert r.set_state('STANDBY_POST')
    assert not r.sb_win and not r.temp_hist


def test_local_socket_permissions_expiration_exclusivity_and_cleanup(tmp_path):
    path=tmp_path/'control.sock'
    control=ControlSocket(path)
    try:
        assert path.stat().st_mode & 0o777==0o600
        with pytest.raises(BlockingIOError):
            ControlSocket(path)
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as client:
            client.bind(str(tmp_path/'client.sock')); client.settimeout(1)
            client.sendto(json.dumps({'command':'select','motion':'idle','sent':time.monotonic()-2}).encode(),str(path))
            assert control.poll() is None
            response=json.loads(client.recv(4096))
            assert not response['ok'] and 'expired' in response['message']
            client.sendto(json.dumps({'command':'select','motion':'idle','sent':time.monotonic()}).encode(),str(path))
            packet,address=control.poll()
            assert packet['motion']=='idle'
            control.reply(address,{'ok':True})
            assert json.loads(client.recv(4096))['ok']
            client.sendto(b'not json',str(path))
            assert control.poll() is None
            assert not json.loads(client.recv(4096))['ok']
    finally:
        control.close()
    assert not path.exists()
    again=ControlSocket(path)
    again.close()


def test_control_socket_does_not_remove_non_socket_path(tmp_path):
    path=tmp_path/'control.sock'; path.write_text('important')
    with pytest.raises(ValueError,match='non-socket'):
        ControlSocket(path)
    assert path.read_text()=='important'


def test_select_sent_during_playback_is_rejected_even_if_polled_in_post(cfg,tmp_path):
    r=runner(cfg)
    path=tmp_path/'control.sock'
    r.control=ControlSocket(path)
    r.bank_aliases=['idle','walk']
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as client:
            client.bind(str(tmp_path/'client.sock')); client.settimeout(1)
            # Request was made in the preceding state; POST began afterwards.
            sent=time.monotonic()
            r.state_t=sent+.001
            client.sendto(json.dumps({'command':'select','motion':'walk','sent':sent}).encode(),str(path))
            r.poll_policy_control(state(),sent+.002)
            answer=json.loads(client.recv(4096))
            assert not answer['ok'] and 'not queued' in answer['message']
            assert r.switching is None
            r.poll_policy_control(state(),sent+.02)
            assert r.switching is None
    finally:
        r.control.close()


def test_unknown_motion_does_not_interrupt_balancing(cfg,tmp_path):
    r=runner(cfg); old=r.core
    r.bank_aliases=['idle','walk']
    path=tmp_path/'control.sock'; r.control=ControlSocket(path)
    try:
        with socket.socket(socket.AF_UNIX,socket.SOCK_DGRAM) as client:
            client.bind(str(tmp_path/'client.sock')); client.settimeout(1)
            client.sendto(json.dumps({'command':'select','motion':'not-a-motion','sent':time.monotonic()}).encode(),str(path))
            r.poll_policy_control(state(),time.monotonic())
            assert not json.loads(client.recv(4096))['ok']
            assert r.core is old and r.state=='STANDBY_POST' and r.switching is None
    finally:
        r.control.close()


def test_handoff_reference_has_future_pose_lookahead_and_matching_velocity(cfg):
    old=DeployCore(cfg,policy(.1),clip('a'))
    target=DeployCore(cfg,policy(.1),clip('b',.2))
    handoff=PolicyHandoff(old,target,2)
    ref=handoff.reference
    assert ref.num_frames==101
    np.testing.assert_allclose(ref.joint_pos[0],0)
    np.testing.assert_allclose(ref.joint_pos[-1],.2)
    assert np.all(np.diff(ref.joint_pos[:,0])>=0)
    np.testing.assert_allclose(ref.goal[[0,-1],29:58],0,atol=1e-10)
    velocity=np.gradient(ref.joint_pos[:,0],cfg.control_dt)
    np.testing.assert_allclose(ref.goal[1:-1,29],velocity[1:-1],atol=1e-4)
    step(handoff)
    future=old.goal().reshape(10,68)
    assert np.all(np.diff(future[:,0])>0)


def test_cross_model_runner_commit_installs_target_callable(cfg):
    from supermimic.deploy.live_switch import Motion
    r=runner(cfg)
    target_fn=policy(.3)
    r.bank_aliases=['idle','walk']
    r.bank=SimpleNamespace(motions={
        'idle':Motion('idle','model_a',r.clips[0],r.policy_fn),
        'walk':Motion('walk','model_b',r.clips[1],target_fn),
    })
    assert r.begin_policy_switch(1,state(),time.monotonic())[0]
    while not r.switching.done:
        step(r.switching)
    assert r.finish_policy_switch()
    assert r.policy_fn is target_fn and r.clip_idx==1
    np.testing.assert_allclose(step(r.core).action,.3)
    assert r.state=='STANDBY_PRE'


def test_actual_control_client_can_select_and_query_without_dds(cfg,tmp_path):
    from policy_control import request
    r=runner(cfg)
    # Selecting the same alias exercises replay without a bank/model load.
    r.bank_aliases=['idle','walk']
    path=tmp_path/'control.sock'; r.control=ControlSocket(path)
    answer=[]
    try:
        client=threading.Thread(target=lambda:answer.append(request(path,'select','idle')))
        client.start()
        deadline=time.monotonic()+1
        while not answer and time.monotonic()<deadline:
            r.poll_policy_control(state(),time.monotonic())
            time.sleep(.001)
        client.join(timeout=1)
        assert answer and answer[0]['ok'] and answer[0]['state']=='STANDBY_POST'
        assert r.switching is not None
        while not r.switching.done:
            step(r.switching)
        r.finish_policy_switch()
        assert r.state=='STANDBY_PRE'
    finally:
        r.control.close()


@pytest.mark.parametrize('ignore_temperature_safety',[False,True])
def test_live_runner_constructor_and_shutdown_without_dds(tmp_path,ignore_temperature_safety):
    import subprocess
    root=Path(__file__).resolve().parents[1]
    manifest=root.parent/'g1_deploy/motion_suite/motions.json'
    if not manifest.exists():
        pytest.skip('local prepared motion bank is not available')
    receipt=tmp_path/'receipt.json'
    code=f'''
import json,sys
from pathlib import Path
sys.path.insert(0,{str(BRIDGE)!r})
sys.path.insert(0,{str(BRIDGE.parent)!r})
import g1_runner_bridge as runner

def offline_startup(self):
    assert self.a.ignore_temperature_safety is {ignore_temperature_safety!r}
    assert self.bank_aliases[self.clip_idx]=='clap'
    assert self.policy_fn is self.bank.motions['clap'].policy
    assert self.core is None and not self.cmd_started
    assert self.control.path.exists()
    Path({str(receipt)!r}).write_text(json.dumps({{"initial":"clap","preloaded":self.bank_aliases}}))

def offline_run(self):
    assert self.bridge is None and not self.cmd_started
runner.Runner.startup=offline_startup
runner.Runner.run=offline_run
runner.main()
'''
    result=subprocess.run([sys.executable,'-c',code,'--iface','lo','--domain','1','--sim','--stage','ground',
                           '--artifacts',str(root/'deploy_artifacts/e11c_0220000'),
                           '--clip',str(root/'deploy_artifacts/clips/shoulder_clap_004__A041_M.npz'),
                           '--policy-bank',str(manifest),'--policy-bank-root',str(root/'deploy_artifacts'),
                           '--policy-control-socket',str(tmp_path/'control.sock'),
                           '--log-dir',str(tmp_path/'logs'),'--cmd-log-max-s','1','--rec-max-s','1']
                          + (['--ignore-temperature-safety'] if ignore_temperature_safety else []),
                          capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stdout+result.stderr
    data=json.loads(receipt.read_text())
    assert data['initial']=='clap' and data['preloaded']==list(json.loads(manifest.read_text()))
    assert not (tmp_path/'control.sock').exists()


@pytest.fixture
def thermal_runner(cfg):
    r=Runner.__new__(Runner)
    r.cfg=cfg
    r.a=SimpleNamespace(thermal_stop_temp=65.,max_temp=80.,temp_rate_min_s=10.,
                        ignore_heating_rate=False,no_temperature_damp=False,ignore_temperature_safety=False,
                        standby_min_s=3.,standby_window_s=3.,standby_max_tilt_deg=8.,
                        standby_max_dq=1.,warn_temp=70.,sim_band_file=None,
                        thermal_stop_pose='current',thermal_stop_s=1.)
    r.live=np.ones(29,bool); r.temp_hr=np.zeros((29,2))
    r.overtemp_warned=False; r.temp_hist=deque()
    r.events=[]; r.state='POLICY'
    r.event=lambda kind,msg:r.events.append((kind,msg))
    def damp(reason):
        r.state='DAMP'; r.events.append(('DAMP',reason))
    r.damp=damp
    r.clips=[clip('idle')]; r.clip_idx=0; r.standby_ok=[True]
    r.static_pre=[static_clip(r.clips[0],0)]
    r.sb_win=deque((float(t),0.,0.) for t in np.linspace(0,3,151))
    return r


def test_heating_rate_default_retains_history_requirement(thermal_runner):
    r=thermal_runner
    r.temp_hist.append((0.,np.full(29,50.)))
    ok,why=r.temp_forecast(SimpleNamespace(num_frames=3000))
    assert not ok and 'heating rate needs' in why


def test_ignore_heating_rate_skips_prediction_and_minimum_history(thermal_runner):
    r=thermal_runner
    r.temp_hist=deque((float(t),np.full(29,40+t)) for t in range(21))
    motion=SimpleNamespace(num_frames=3000)
    assert not r.temp_forecast(motion)[0]  # predicted end temperature is above 65
    r.a.ignore_heating_rate=True
    ok,why=r.temp_forecast(motion)
    assert ok and 'heating rate ignored' in why
    r.temp_hist=deque([(0.,np.full(29,60.))])
    assert r.temp_forecast(motion)[0]  # no rate-history wait


@pytest.mark.parametrize('temperature',[65.,70.,81.])
def test_ignore_rate_does_not_allow_start_at_or_above_current_thermal_limit(thermal_runner,temperature):
    r=thermal_runner; r.a.ignore_heating_rate=True
    r.a.no_temperature_damp=True
    r.temp_hist=deque([(0.,np.full(29,temperature))])
    assert not r.temp_forecast(r.clips[0])[0]


def test_ignore_rate_needs_a_valid_current_temperature_sample(thermal_runner):
    r=thermal_runner; r.a.ignore_heating_rate=True
    assert not r.temp_forecast(r.clips[0])[0]
    r.temp_hist=deque([(0.,np.full(29,np.nan))])
    assert not r.temp_forecast(r.clips[0])[0]


def test_warning_temperature_start_limit_still_applies_with_both_flags(thermal_runner):
    r=thermal_runner; r.a.ignore_heating_rate=True; r.a.no_temperature_damp=True
    st=state(); st.temp[:]=75
    ok,why=r.standby_allowed(20,st)
    assert not ok and '--warn-temp' in why


def test_max_temperature_still_damps_by_default(thermal_runner):
    r=thermal_runner
    st=state(); st.temp[2,1]=80.
    assert not r.check_temperature_damp(st)
    st.temp[2,1]=81.
    assert r.check_temperature_damp(st)
    assert r.state=='DAMP' and r.events[-1][0]=='DAMP'


def test_no_temperature_damp_warns_without_damping_and_rearms_warning(thermal_runner):
    r=thermal_runner; r.a.no_temperature_damp=True
    st=state(); st.temp[2,1]=81.
    assert not r.check_temperature_damp(st)
    assert r.state=='POLICY' and len(r.events)==1 and r.events[0][0]=='WARN'
    assert '--no-temperature-damp' in r.events[0][1]
    assert not r.check_temperature_damp(st)
    assert len(r.events)==1
    st.temp[2,1]=79.; r.check_temperature_damp(st)
    st.temp[2,1]=81.; r.check_temperature_damp(st)
    assert len(r.events)==2


def test_temperature_headroom_is_preserved(thermal_runner):
    r=thermal_runner; r.temp_hr[2,1]=5.
    st=state(); st.temp[2,1]=84.9
    assert not r.check_temperature_damp(st)
    st.temp[2,1]=85.1
    assert r.check_temperature_damp(st)


def test_controlled_thermal_stop_still_works_with_both_flags(thermal_runner):
    r=thermal_runner; r.a.ignore_heating_rate=True; r.a.no_temperature_damp=True
    core=DeployCore(r.cfg,policy(0),r.clips[0])
    r.thermal_stop(core,r.clips[0],0,65.,0)
    assert r.thermal['motor']==0 and r.thermal['temp']==65.
    assert r.state=='POLICY'
    assert r.events[-1][0]=='WARN' and 'THERMAL STOP' in r.events[-1][1]


@pytest.mark.parametrize('temperature',[65.,67.,68.,81.,100.])
def test_full_temperature_override_allows_hot_switch_and_replay(cfg,thermal_runner,temperature):
    r=runner(cfg)
    r.a.__dict__.update(thermal_runner.a.__dict__)
    r.live=thermal_runner.live; r.temp_hr=thermal_runner.temp_hr
    r.standby_ok=[True,True]
    r.sb_win=thermal_runner.sb_win
    r.temp_hist=deque([(0.,np.full(29,temperature))])
    r.standby_allowed=Runner.standby_allowed.__get__(r)
    r.a.ignore_heating_rate=True
    st=state(); st.temp[:]=temperature
    old=r.core
    assert not r.begin_policy_switch(1,st,time.monotonic())[0]
    assert r.core is old and r.switching is None
    r.a.ignore_temperature_safety=True
    # No temperature history is needed, even when the narrower flag is off.
    r.a.ignore_heating_rate=False
    r.temp_hist.clear()
    assert r.begin_policy_switch(1,st,time.monotonic())[0]
    while not r.switching.done:
        step(r.switching)
    assert r.finish_policy_switch()
    r.state='STANDBY_POST'; r.state_t=time.monotonic()-20
    r.sb_win=deque((float(t),0.,0.) for t in np.linspace(0,3,151))
    assert r.begin_policy_switch(r.clip_idx,st,time.monotonic())[0]


def test_full_temperature_override_bypasses_startup_limits_and_runtime_actions(thermal_runner):
    r=thermal_runner
    st=state(); st.temp[:]=100.
    assert r.temperature_limit_exceeded(st,r.a.max_temp)
    r.a.ignore_temperature_safety=True
    assert not r.temperature_limit_exceeded(st,r.a.max_temp)
    assert not r.temperature_limit_exceeded(st,r.a.warn_temp)
    assert not r.check_temperature_damp(st)
    assert not r.events and r.state=='POLICY'
    core=DeployCore(r.cfg,policy(0),r.clips[0]); original=core.clip
    r.thermal=None
    r.thermal_stop(core,r.clips[0],0,100.,0)
    assert r.thermal is None and core.clip is original and not r.events
    # Keep raw readings visible rather than replacing them with cooler values.
    assert '100' in r.temp_line(st)
    np.testing.assert_array_equal(st.temp,100.)


def test_full_temperature_override_preserves_stability_and_pose_gates(thermal_runner):
    r=thermal_runner; r.a.ignore_temperature_safety=True
    r.a.no_policy=False; r.a.hold_min_s=1.; r.a.hold_tol=.1
    r.a.start_max_dq=.5; r.a.start_max_tilt_deg=10.
    st=state(); st.temp[:]=100.
    assert r.policy_allowed(2.,st,np.zeros(29),r.clips[0])[0]
    assert r.standby_allowed(20.,st)[0]
    st.q[0]=.2
    assert not r.policy_allowed(2.,st,np.zeros(29),r.clips[0])[0]
    r.sb_win[-1]=(3.,0.,2.)
    ok,why=r.standby_allowed(20.,st)
    assert not ok and 'max|dq|' in why


def test_full_temperature_override_skips_forecast_without_valid_samples(thermal_runner):
    r=thermal_runner; r.a.ignore_temperature_safety=True
    assert r.temp_forecast(r.clips[0])[0]
    r.temp_hist=deque([(0.,np.full(29,np.nan))])
    assert r.temp_forecast(r.clips[0])[0]


def test_waist_temperature_override_preserves_position_interlock():
    import g1_common as gc
    st=state(); st.motorstate=np.zeros(29,dtype=np.uint32)
    assert gc.waist_active(st) is not None
    assert gc.waist_active(st,check_temperature=False) is None
    st.q[13]=.03
    assert gc.waist_active(st,check_temperature=False) is not None


def test_full_temperature_override_suppresses_temperature_warnings_only(thermal_runner):
    r=thermal_runner
    r.a.status_s=3.; r.a.warn_leg_dq=1.; r.a.warn_leg_dq_s=.5
    r.a.warn_yaw_deg=15.; r.a.standby_warn_s=120.
    r.sbs=None; r.state_t=0.
    st=state(); st.temp[:]=100.
    r.standby_status('STANDBY_PRE',20.,20.,st,0.,0.)
    assert any(kind=='WARN' and 'temperature' in msg for kind,msg in r.events)
    r.events.clear(); r.sbs=None
    r.a.ignore_temperature_safety=True
    r.standby_status('STANDBY_PRE',121.,121.,st,0.,0.)
    assert not any('temperature' in msg for kind,msg in r.events)
    assert any(kind=='WARN' and 'standby cap' in msg for kind,msg in r.events)
