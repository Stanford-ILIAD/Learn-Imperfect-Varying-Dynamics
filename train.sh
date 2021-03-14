### Reacher Clockwise 
python train_reacher.py --env reacher_custom-action1-v0 --num-epochs 5000 --demo_file_list demos/reacher_custom-action1-v0_0.0.pkl demos/reacher_custom-action2-v0_0.0.pkl demos/reacher_custom-action1-v0_1.0.pkl --log-interval 10 --test_episodes 100 --result_file results/reacher_results/action1-action2-action1-feas_conf.pkl --snapshot_file snapshot/reacher_snapshot/action1-action2-action1-feas_conf.tar  --percent_list 0.05 0.9 0.05 --feasibility --optimality --delta-s 0.001 --sigma 50.

### Reacher Counter-Clockwise
python train_reacher.py --env reacher_custom-action2-v0 --num-epochs 5000 --demo_file_list demos/reacher_custom-action2-v0_0.0.pkl demos/reacher_custom-action1-v0_0.0.pkl demos/reacher_custom-action2-v0_1.0.pkl --log-interval 10 --test_episodes 100 --result_file results/reacher_results/action2-action1-action2-feas_conf.pkl --snapshot_file snapshot/reacher_snapshot/action2-action1-action2-feas_conf.tar  --percent_list 0.05 0.9 0.05 --feasibility --optimality --delta-s 0.001 --sigma 50.

### Driving Fast
python train_driving.py --env ContinuousFastRandom-v0 --num-epochs 3000 --demo_file_list demos/driving-fast.pkl demos/driving-slow.pkl demos/driving-fast-suboptimal.pkl --log-interval 10 --test_episodes 100 --result_file results/driving_results/fast_slow_fast-feas_conf.pkl --snapshot_file snapshot/driving_snapshot/fast_slow_fast-feas_conf.tar --percent_list 0.3 0.60 0.1 --feasibility --optimality --delta-s  --sigma 150.

## Driving Slow
python train_driving.py --env ContinuousSlowRandom-v0 --num-epochs 3000 --demo_file_list demos/driving-slow.pkl demos/driving-fast.pkl demos/driving-slow-suboptimal.pkl --log-interval 10 --test_episodes 100 --result_file results/driving_results/slow_fast_slow-feas_conf.pkl --snapshot_file snapshot/driving_snapshot/slow_fast_slow-feas_conf.tar --percent_list 0.3 0.60 0.1 --feasibility --optimality --delta-s --sigma 150.

