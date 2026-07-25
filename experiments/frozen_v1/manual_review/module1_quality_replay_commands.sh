#!/usr/bin/env bash
set -e

echo '[module1_quality] 1/80 bucket=highest_pass global_segment_id=13074'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/001875_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 2/80 bucket=highest_pass global_segment_id=9809'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/012468_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 3/80 bucket=highest_pass global_segment_id=8308'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0010/Plie_Heel_Lift_To_Arm_Openef_clip_17_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 4/80 bucket=highest_pass global_segment_id=4350'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0097/Side_To_Side_Ab_Squeeze_clip_14_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 5/80 bucket=highest_pass global_segment_id=14473'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0059/Crunch_To_Butt_Kick_R_clip_13_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 6/80 bucket=highest_pass global_segment_id=3517'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/music/Play_the_stringed_guqin_77_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 7/80 bucket=highest_pass global_segment_id=5228'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/009529_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 8/80 bucket=highest_pass global_segment_id=11897'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/001900_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 9/80 bucket=highest_pass global_segment_id=394'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0069/Cross_Crunches_R_Clip1_clip_30_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 10/80 bucket=highest_pass global_segment_id=11444'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/014242_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 11/80 bucket=highest_pass global_segment_id=7179'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/012801_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 12/80 bucket=highest_pass global_segment_id=18569'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/000764_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 13/80 bucket=highest_pass global_segment_id=13810'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/003321_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 14/80 bucket=highest_pass global_segment_id=13693'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0095/Reach_To_Crunch_L_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 15/80 bucket=highest_pass global_segment_id=13214'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0031/Jump_Iniout_To_Cross_Crunchfs_clip_2_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 16/80 bucket=highest_pass global_segment_id=9711'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Zhan_Zhuang_Gong_while_walking_1_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 17/80 bucket=highest_pass global_segment_id=16976'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/EgoBody/recording_20210921_S10_S11_02/body_idx_0/004_chunk_0004.npz --start_frame 150 --end_frame_exclusive 192 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 18/80 bucket=highest_pass global_segment_id=2543'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/006407_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 19/80 bucket=highest_pass global_segment_id=4497'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0081/Chair_Twist_clip_8_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 20/80 bucket=highest_pass global_segment_id=20842'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Walking_forward_during_standing_1_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 21/80 bucket=near_pass_borderline global_segment_id=18154'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/005205_chunk_0000.npz --start_frame 150 --end_frame_exclusive 190 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 22/80 bucket=near_pass_borderline global_segment_id=2803'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_while_Putting_it_down_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 23/80 bucket=near_pass_borderline global_segment_id=18153'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/005205_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 24/80 bucket=near_pass_borderline global_segment_id=6442'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_during_Digging_1_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 25/80 bucket=near_pass_borderline global_segment_id=1097'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/kungfu/Shaolin_Kung_Fu_Wushu_Basic_Tiger_Sword_Training_1_clip3_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 26/80 bucket=near_pass_borderline global_segment_id=13994'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Rolling_Downhill_while_walking_1_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 27/80 bucket=near_pass_borderline global_segment_id=6443'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_during_Digging_1_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 28/80 bucket=near_pass_borderline global_segment_id=12764'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Bow_your_head_and_walking_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 29/80 bucket=near_pass_borderline global_segment_id=12862'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Stomping_and_walking_1_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 30/80 bucket=near_pass_borderline global_segment_id=16058'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/animation/Ways_to_Jump_+_Sit_+_Fall_Expressive_clip1_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 122 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 31/80 bucket=near_pass_borderline global_segment_id=15195'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/000605_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 32/80 bucket=near_pass_borderline global_segment_id=17929'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_and_Listen_to_music_at_the_same_time_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 33/80 bucket=near_pass_borderline global_segment_id=12354'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/EgoBody/recording_20220318_S33_S34_02/body_idx_1/000_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 34/80 bucket=near_pass_borderline global_segment_id=21495'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/001917_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 35/80 bucket=near_pass_borderline global_segment_id=16869'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Simultaneously_walking_and_Tug-of-War_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 36/80 bucket=near_pass_borderline global_segment_id=18193'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/002113_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 37/80 bucket=near_pass_borderline global_segment_id=6790'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_while_squeeze_fingers_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 38/80 bucket=near_pass_borderline global_segment_id=20201'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/001184_chunk_0001.npz --start_frame 100 --end_frame_exclusive 132 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 39/80 bucket=near_pass_borderline global_segment_id=13067'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/aist/subset_0002/Dance_Ballet_Jazz_Passe_clip_6_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 40/80 bucket=near_pass_borderline global_segment_id=18042'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/001184_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 41/80 bucket=near_borderline_reject global_segment_id=910'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/GRAB/s8/mouse_pass_1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 42/80 bucket=near_borderline_reject global_segment_id=13993'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Rolling_Downhill_while_walking_1_clip1_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 43/80 bucket=near_borderline_reject global_segment_id=8694'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_and_Kicking_a_football_to_shoot_at_the_same_time_1_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 44/80 bucket=near_borderline_reject global_segment_id=3933'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/music/beat_drums_and_gongs_32_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 45/80 bucket=near_borderline_reject global_segment_id=19271'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/haa500/fire_breathing_4_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 46/80 bucket=near_borderline_reject global_segment_id=15127'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/011736_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 47/80 bucket=near_borderline_reject global_segment_id=6360'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_while_Grasping_Objects_clip1_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 48/80 bucket=near_borderline_reject global_segment_id=14269'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Hitting_with_an_Object_while_walking_clip1_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 49/80 bucket=near_borderline_reject global_segment_id=18631'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/kungfu/Aerial_Kick_Kungfu_wushu_21_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 50/80 bucket=near_borderline_reject global_segment_id=18261'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/perform/carry_towels_three_times_clip1_clip1_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 51/80 bucket=near_borderline_reject global_segment_id=11'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/GRAB/s10/spheremedium_pass_1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 52/80 bucket=near_borderline_reject global_segment_id=15558'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Holding_things_in_your_hands_during_walking_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 53/80 bucket=near_borderline_reject global_segment_id=6445'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_during_Digging_1_clip1_chunk_0001.npz --start_frame 150 --end_frame_exclusive 200 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 54/80 bucket=near_borderline_reject global_segment_id=9692'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Ice_Skating_while_standing_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 55/80 bucket=near_borderline_reject global_segment_id=14403'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/EgoBody/recording_20220315_S30_S21_02/body_idx_1/004_chunk_0002.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 56/80 bucket=near_borderline_reject global_segment_id=100'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/custom/squat_chunk_0019.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 57/80 bucket=near_borderline_reject global_segment_id=10021'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0063/Narrow_Squat_To_Star_Rfach_clip_4_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 58/80 bucket=near_borderline_reject global_segment_id=9'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/GRAB/s10/spheremedium_pass_1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 59/80 bucket=near_borderline_reject global_segment_id=20267'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/GRAB/s3/toothpaste_pass_1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 60/80 bucket=near_borderline_reject global_segment_id=1182'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/007104_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 61/80 bucket=lowest_reject global_segment_id=5348'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Swinging_during_standing_1_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 62/80 bucket=lowest_reject global_segment_id=10106'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/kungfu/_Form_Tai_Chi_Demonstration_Master_form24_the_golden_rooster_stands_on_one_leg_clip1_chunk_0001.npz --start_frame 150 --end_frame_exclusive 174 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 63/80 bucket=lowest_reject global_segment_id=3482'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/fitness/subset_0098/Standing_Split_Pulse_To_Sout_R_clip_1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 64/80 bucket=lowest_reject global_segment_id=9732'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/001917_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 65/80 bucket=lowest_reject global_segment_id=15812'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/EgoBody/recording_20220312_S28_S29_04/body_idx_1/001_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 66/80 bucket=lowest_reject global_segment_id=8240'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/haa500/badminton_overswing_16_clip2_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 67/80 bucket=lowest_reject global_segment_id=6424'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Amusement_Park_Roller_Coaster_and_standing_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 68/80 bucket=lowest_reject global_segment_id=5347'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Swinging_during_standing_1_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 69/80 bucket=lowest_reject global_segment_id=7728'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/GRAB/s10/cylinderlarge_inspect_1_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 70/80 bucket=lowest_reject global_segment_id=10503'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Goal_Shot_during_standing_2_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 71/80 bucket=lowest_reject global_segment_id=8493'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/perform/cook_clip8_chunk_0003.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 72/80 bucket=lowest_reject global_segment_id=13831'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Diving_and_walking_at_the_same_time_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 73/80 bucket=lowest_reject global_segment_id=7246'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/perform/physical_display_woman_clip4_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 74/80 bucket=lowest_reject global_segment_id=15592'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/music/Play_Dulcimer_31_clip1_chunk_0002.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 75/80 bucket=lowest_reject global_segment_id=6361'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/walking_while_Grasping_Objects_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 76/80 bucket=lowest_reject global_segment_id=4347'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/music/Play_the_violin_11_clip3_chunk_0002.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 77/80 bucket=lowest_reject global_segment_id=1270'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/humanml/007562_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 78/80 bucket=lowest_reject global_segment_id=284'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/haa500/eat_spagetti_11_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 79/80 bucket=lowest_reject global_segment_id=8241'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/haa500/badminton_overswing_16_clip2_chunk_0000.npz --start_frame 50 --end_frame_exclusive 64 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module1_quality] 80/80 bucket=lowest_reject global_segment_id=5349'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file PHUMA_wbt_motions/g1_all/idea400/Swinging_during_standing_1_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 127 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

