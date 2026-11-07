# Source this file before deployment or simulation.
export G1_DEPLOY_HOME=/home/yiannis/policies/g1_deploy
export PATH="$G1_DEPLOY_HOME/venv/bin:$PATH"
export CYCLONEDDS_HOME="$G1_DEPLOY_HOME/cyclonedds"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$G1_DEPLOY_HOME/venv/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export G1_UNITREE_XML="$G1_DEPLOY_HOME/unitree_mujoco/unitree_robots/g1/scene_29dof.xml"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
unset PYTHONPATH
