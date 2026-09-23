#!/usr/bin/env bash
set -e

echo '[module2_difficulty] 1/236 bucket=lowest global_segment_id=2069'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0026/Pause_Worried_clip_6_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 2/236 bucket=lowest global_segment_id=1122'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/dance2/subject4_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 3/236 bucket=lowest global_segment_id=1124'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/dance2/subject4_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 4/236 bucket=lowest global_segment_id=18060'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0006/Emotion_Be_clip_5_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 5/236 bucket=lowest global_segment_id=2068'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0026/Pause_Worried_clip_6_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 6/236 bucket=lowest global_segment_id=1123'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/dance2/subject4_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 7/236 bucket=lowest global_segment_id=13808'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/003321_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 8/236 bucket=lowest global_segment_id=18887'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/008098_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 9/236 bucket=lowest global_segment_id=17819'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/000479_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 10/236 bucket=lowest global_segment_id=16007'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/006902_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 11/236 bucket=lowest global_segment_id=3477'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0000/Close-up_Change_clip_4_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 12/236 bucket=lowest global_segment_id=10538'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/005394_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 13/236 bucket=lowest global_segment_id=6094'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0026/Pause_Worried_clip_9_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 14/236 bucket=lowest global_segment_id=10537'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/005394_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 15/236 bucket=lowest global_segment_id=2070'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0026/Pause_Worried_clip_6_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 16/236 bucket=lowest global_segment_id=12859'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0034/Hand_Clapping_Choreography_clip_9_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 17/236 bucket=lowest global_segment_id=6118'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/003091_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 18/236 bucket=lowest global_segment_id=9324'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0026/Pause_Chubby_clip_1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 19/236 bucket=lowest global_segment_id=17820'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/000479_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 20/236 bucket=lowest global_segment_id=6093'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0026/Pause_Worried_clip_9_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 21/236 bucket=highest global_segment_id=17999'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0054.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 22/236 bucket=highest global_segment_id=13961'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/sprint1/subject4_chunk_0076.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 23/236 bucket=highest global_segment_id=12640'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/jumps1/subject1_chunk_0062.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 24/236 bucket=highest global_segment_id=14571'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run2/subject1_chunk_0016.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 25/236 bucket=highest global_segment_id=78'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0053.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 26/236 bucket=highest global_segment_id=16369'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject5_chunk_0009.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 27/236 bucket=highest global_segment_id=1095'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/kungfu/Shaolin_Kung_Fu_Wushu_Basic_Tiger_Sword_Training_1_clip3_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 28/236 bucket=highest global_segment_id=9937'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/sprint1/subject2_chunk_0009.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 29/236 bucket=highest global_segment_id=18011'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0033.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 30/236 bucket=highest global_segment_id=1455'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/sprint1/subject2_chunk_0028.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 31/236 bucket=highest global_segment_id=20160'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run2/subject4_chunk_0056.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 32/236 bucket=highest global_segment_id=17997'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0054.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 33/236 bucket=highest global_segment_id=20162'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run2/subject4_chunk_0056.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 34/236 bucket=highest global_segment_id=9219'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/dance1/subject2_chunk_0019.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 35/236 bucket=highest global_segment_id=13870'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/aist/subset_0004/Dance_Street_Jazz_Positions_Des_Bras_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 36/236 bucket=highest global_segment_id=18639'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run2/subject4_chunk_0061.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 37/236 bucket=highest global_segment_id=8888'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/002558_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 38/236 bucket=highest global_segment_id=14572'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run2/subject1_chunk_0016.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 39/236 bucket=highest global_segment_id=13963'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/sprint1/subject4_chunk_0076.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 40/236 bucket=highest global_segment_id=75'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0053.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 41/236 bucket=bin_0_random global_segment_id=19068'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/007944_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 42/236 bucket=bin_0_random global_segment_id=13239'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Suona_1_clip1_chunk_0002.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 43/236 bucket=bin_0_random global_segment_id=20836'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/005171_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 44/236 bucket=bin_0_random global_segment_id=17818'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/000479_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 45/236 bucket=bin_0_random global_segment_id=18946'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/014271_chunk_0001.npz --start_frame 50 --end_frame_exclusive 95 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 46/236 bucket=bin_0_random global_segment_id=13262'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/006893_chunk_0000.npz --start_frame 150 --end_frame_exclusive 167 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 47/236 bucket=bin_0_random global_segment_id=19911'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Ruan_60_clip1_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 48/236 bucket=bin_0_random global_segment_id=18492'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/012438_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 49/236 bucket=bin_0_random global_segment_id=18060'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0006/Emotion_Be_clip_5_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 50/236 bucket=bin_0_random global_segment_id=1619'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Suona_48_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 51/236 bucket=bin_1_random global_segment_id=9053'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/008085_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 52/236 bucket=bin_1_random global_segment_id=10687'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Ancient_Drum_9_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 52 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 53/236 bucket=bin_1_random global_segment_id=18022'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/004647_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 54/236 bucket=bin_1_random global_segment_id=9555'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/003276_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 55/236 bucket=bin_1_random global_segment_id=12415'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/004991_chunk_0002.npz --start_frame 50 --end_frame_exclusive 55 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 56/236 bucket=bin_1_random global_segment_id=19338'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/perform/Writing_with_brush_hold_a_pen_12_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 57/236 bucket=bin_1_random global_segment_id=15874'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Bass_25_clip5_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 58/236 bucket=bin_1_random global_segment_id=20424'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Dulcimer_32_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 59/236 bucket=bin_1_random global_segment_id=12684'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/004863_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 60/236 bucket=bin_1_random global_segment_id=19659'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0025/Move_Walk_clip_28_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 61/236 bucket=bin_2_random global_segment_id=16643'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/custom/turn_chunk_0007.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 62/236 bucket=bin_2_random global_segment_id=1226'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/standing_and_Bartending_at_the_same_time_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 63/236 bucket=bin_2_random global_segment_id=2122'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/005998_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 64/236 bucket=bin_2_random global_segment_id=9151'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0008/Heel_Lift_Walk_Inhale_Exhale_chunk_0004.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 65/236 bucket=bin_2_random global_segment_id=4684'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0034/Hand_Clapping_Choreography_clip_3_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 66/236 bucket=bin_2_random global_segment_id=21556'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0018/Chest_To_Back_Opener_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 67/236 bucket=bin_2_random global_segment_id=5448'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/006751_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 68/236 bucket=bin_2_random global_segment_id=17748'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/012744_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 69/236 bucket=bin_2_random global_segment_id=5086'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0010/Pilates_Knee_Strike_R_clip_6_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 70/236 bucket=bin_2_random global_segment_id=8266'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/013032_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 71/236 bucket=bin_3_random global_segment_id=14898'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0091/Calf_Raise_To_Squat_clip_7_chunk_0000.npz --start_frame 150 --end_frame_exclusive 179 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 72/236 bucket=bin_3_random global_segment_id=19675'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0008/Heel_Lift_Walk_Inhale_Exhale_chunk_0003.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 73/236 bucket=bin_3_random global_segment_id=8702'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0007/Emotion_Good_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 74/236 bucket=bin_3_random global_segment_id=6960'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0108/Single_Leg_Lift_Shouidfr_Prfss_R_clip_14_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 75/236 bucket=bin_3_random global_segment_id=3469'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Sandhammer_5_clip1_chunk_0002.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 76/236 bucket=bin_3_random global_segment_id=12118'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/002636_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 77/236 bucket=bin_3_random global_segment_id=996'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/005878_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 78/236 bucket=bin_3_random global_segment_id=12958'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Hand_touches_neck_while_walking_1_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 125 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 79/236 bucket=bin_3_random global_segment_id=19039'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/EgoBody/recording_20210921_S10_S11_02/body_idx_0/002_chunk_0004.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 80/236 bucket=bin_3_random global_segment_id=17758'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0047/Standing_Butt_Pulses_R_clip_14_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 81/236 bucket=bin_4_random global_segment_id=19773'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0104/Lean_Forward_Back_Pulses_clip_12_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 82/236 bucket=bin_4_random global_segment_id=2686'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Toes_touching_during_standing_1_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 83/236 bucket=bin_4_random global_segment_id=14642'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/aist/subset_0003/Dance_Street_Jazz_Jump_clip_18_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 84/236 bucket=bin_4_random global_segment_id=1494'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0011/Rainbow_To_Cross_Crunch_F_clip_10_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 85/236 bucket=bin_4_random global_segment_id=1978'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/012766_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 86/236 bucket=bin_4_random global_segment_id=19813'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/009732_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 87/236 bucket=bin_4_random global_segment_id=14188'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/006463_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 88/236 bucket=bin_4_random global_segment_id=6400'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Basking_in_the_sun_and_walking_at_the_same_time_1_clip1_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 89/236 bucket=bin_4_random global_segment_id=19823'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Acting_like_a_baby_while_standing_1_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 90/236 bucket=bin_4_random global_segment_id=18909'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0017/Balancing_Leg_Press_R_clip_30_chunk_0002.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 91/236 bucket=bin_5_random global_segment_id=7180'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/012801_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 92/236 bucket=bin_5_random global_segment_id=12407'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/001008_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 93/236 bucket=bin_5_random global_segment_id=11192'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/002873_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 94/236 bucket=bin_5_random global_segment_id=18912'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0022/Pike_Toe_Touch_To_Release_clip_24_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 95/236 bucket=bin_5_random global_segment_id=3669'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/haa500/kick_jianzi_1_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 69 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 96/236 bucket=bin_5_random global_segment_id=17674'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_bass_drum_25_clip1_chunk_0002.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 97/236 bucket=bin_5_random global_segment_id=6942'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/EgoBody/recording_20220312_S28_S29_03/body_idx_1/000_chunk_0003.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 98/236 bucket=bin_5_random global_segment_id=6039'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_and_Spread_your_hands_at_the_same_time_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 99/236 bucket=bin_5_random global_segment_id=18248'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/perform/quit_smoking_clip2_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 100/236 bucket=bin_5_random global_segment_id=19839'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/EgoBody/recording_20210910_S06_S05_03/body_idx_1/000_chunk_0004.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 101/236 bucket=bin_6_random global_segment_id=12512'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Hand_touches_neck_while_walking_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 102/236 bucket=bin_6_random global_segment_id=15014'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/sprint1/subject4_chunk_0012.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 103/236 bucket=bin_6_random global_segment_id=6561'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0071/Heel_Tap_To_Cross_Crunch_clip_25_chunk_0000.npz --start_frame 100 --end_frame_exclusive 135 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 104/236 bucket=bin_6_random global_segment_id=133'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/010512_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 105/236 bucket=bin_6_random global_segment_id=21111'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Rock_Climbing_during_standing_1_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 106/236 bucket=bin_6_random global_segment_id=10986'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0025/Move_Witch_clip_8_chunk_0000.npz --start_frame 150 --end_frame_exclusive 200 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 107/236 bucket=bin_6_random global_segment_id=7421'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0040/Curl_To_Extend_R_clip_12_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 108/236 bucket=bin_6_random global_segment_id=18183'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0037/Step_Out_Jump_Side_Front_Push_clip_5_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 109/236 bucket=bin_6_random global_segment_id=13057'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0002/Dance_Groovy_clip_2_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 110/236 bucket=bin_6_random global_segment_id=19465'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/standing_while_Gestures_in_the_Air_clip1_chunk_0001.npz --start_frame 150 --end_frame_exclusive 174 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 111/236 bucket=bin_7_random global_segment_id=4432'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_and_Lick_at_the_same_time_1_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 112/236 bucket=bin_7_random global_segment_id=19103'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0076/Side_Leg_Lifts_R_clip_33_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 113/236 bucket=bin_7_random global_segment_id=11090'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0081/Cross_Crunches_To_Reach_Up_clip_26_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 114/236 bucket=bin_7_random global_segment_id=16525'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_while_Rub_your_hands_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 115/236 bucket=bin_7_random global_segment_id=15677'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0079/Walk_Into_Plie_clip_13_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 116/236 bucket=bin_7_random global_segment_id=13779'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0017/Kick_I_clip_9_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 117/236 bucket=bin_7_random global_segment_id=14578'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Punching_or_Slapping_and_walking_at_the_same_time_1_clip1_chunk_0001.npz --start_frame 200 --end_frame_exclusive 217 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 118/236 bucket=bin_7_random global_segment_id=11028'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0000/Perform_Ballet_clip_31_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 119/236 bucket=bin_7_random global_segment_id=10281'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0045/Side_Back_Runner_R_clip_11_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 120/236 bucket=bin_7_random global_segment_id=9516'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/fight1/subject3_chunk_0067.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 121/236 bucket=bin_8_random global_segment_id=15997'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0046/Side_Reach_Up_Down_R_clip_2_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 122/236 bucket=bin_8_random global_segment_id=17992'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0110/Xpush_clip_1_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 123/236 bucket=bin_8_random global_segment_id=2269'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/002460_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 124/236 bucket=bin_8_random global_segment_id=12915'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/002754_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 125/236 bucket=bin_8_random global_segment_id=18056'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/012306_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 126/236 bucket=bin_8_random global_segment_id=9821'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0031/Short_Weapon_S_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 127/236 bucket=bin_8_random global_segment_id=11264'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/011098_chunk_0000.npz --start_frame 150 --end_frame_exclusive 184 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 128/236 bucket=bin_8_random global_segment_id=19498'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0017/Yogi_Squat_To_Hamstring_Stretch_clip_26_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 129/236 bucket=bin_8_random global_segment_id=16741'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/009834_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 130/236 bucket=bin_8_random global_segment_id=3182'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/aist/subset_0002/Dance_Ballet_Jazz_clip_7_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 131/236 bucket=bin_9_random global_segment_id=6075'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0086/Side_To_Side_Slam_Jump_clip_12_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 132/236 bucket=bin_9_random global_segment_id=4244'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0048/Step_Out_Knee_Tuck_Punches_R_Clip1_clip_19_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 133/236 bucket=bin_9_random global_segment_id=3377'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0032/Next_Water_Break_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 134/236 bucket=bin_9_random global_segment_id=7992'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/002276_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 135/236 bucket=bin_9_random global_segment_id=11913'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LocoMuJoCo/run_chunk_0068.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 136/236 bucket=bin_9_random global_segment_id=20161'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run2/subject4_chunk_0056.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 137/236 bucket=bin_9_random global_segment_id=17115'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0008/Fighting_Combo_Gehirn_clip_5_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 138/236 bucket=bin_9_random global_segment_id=9992'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0021/Long_Weapon_Umliegendes_clip_9_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 139/236 bucket=bin_9_random global_segment_id=406'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/jumps1/subject2_chunk_0040.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 140/236 bucket=bin_9_random global_segment_id=6645'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/jumps1/subject2_chunk_0043.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 141/236 bucket=boundary_1 global_segment_id=12850'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0034/Other_Be_clip_14_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 142/236 bucket=boundary_1 global_segment_id=19094'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Simultaneously_standing_and_hands_behind_your_back_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 143/236 bucket=boundary_1 global_segment_id=13925'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humman/Wrist_Wrist_Circles_0_clip2_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 144/236 bucket=boundary_1 global_segment_id=14158'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Clarinet_26_clip1_chunk_0003.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 145/236 bucket=boundary_2 global_segment_id=20673'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/007010_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 146/236 bucket=boundary_2 global_segment_id=2595'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/perform/perform_5_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 147/236 bucket=boundary_2 global_segment_id=11095'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Ruan_65_clip1_chunk_0004.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 148/236 bucket=boundary_2 global_segment_id=15485'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/012375_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 149/236 bucket=boundary_3 global_segment_id=2111'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/EgoBody/recording_20220312_S28_S29_04/body_idx_0/003_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 150/236 bucket=boundary_3 global_segment_id=17933'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/009910_chunk_0000.npz --start_frame 150 --end_frame_exclusive 175 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 151/236 bucket=boundary_3 global_segment_id=8170'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Stopping_and_standing_at_the_same_time_2_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 152/236 bucket=boundary_3 global_segment_id=21413'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/010384_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 153/236 bucket=boundary_4 global_segment_id=15979'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/kungfu/Kung_Fu_Nunchucks_Training_Best_Nunchaku_13_clip1_chunk_0004.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 154/236 bucket=boundary_4 global_segment_id=10104'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/kungfu/_Form_Tai_Chi_Demonstration_Master_form24_the_golden_rooster_stands_on_one_leg_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 155/236 bucket=boundary_4 global_segment_id=4329'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0005/Arabesque_Crunch_L_clip_6_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 156/236 bucket=boundary_4 global_segment_id=6773'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0027/Projectile_Slide_clip_5_chunk_0002.npz --start_frame 0 --end_frame_exclusive 49 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 157/236 bucket=boundary_5 global_segment_id=17764'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/010928_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 158/236 bucket=boundary_5 global_segment_id=11647'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/000981_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 159/236 bucket=boundary_5 global_segment_id=8182'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Concession_and_walking_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 160/236 bucket=boundary_5 global_segment_id=5658'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Piloting_and_standing_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 161/236 bucket=boundary_6 global_segment_id=5548'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_during_Back_Pain_2_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 162/236 bucket=boundary_6 global_segment_id=18488'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/006509_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 163/236 bucket=boundary_6 global_segment_id=13909'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Bowling_while_standing_1_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 164/236 bucket=boundary_6 global_segment_id=13947'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Hair_Tossing_and_walking_clip1_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 165/236 bucket=boundary_7 global_segment_id=14913'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/000186_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 166/236 bucket=boundary_7 global_segment_id=10176'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/000216_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 167/236 bucket=boundary_7 global_segment_id=12005'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_while_Crossed_Arms_1_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 168/236 bucket=boundary_7 global_segment_id=10186'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0036/Squat_Side_Up_Punch_clip_6_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 169/236 bucket=boundary_8 global_segment_id=11183'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/013218_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 170/236 bucket=boundary_8 global_segment_id=11873'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/009916_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 171/236 bucket=boundary_8 global_segment_id=8293'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/EgoBody/recording_20210921_S10_S11_02/body_idx_1/002_chunk_0004.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 172/236 bucket=boundary_8 global_segment_id=452'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0014/Single_Knee_Strike_R_clip_22_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 173/236 bucket=boundary_9 global_segment_id=9561'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0108/Small_Arm_Circles_clip_3_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 174/236 bucket=boundary_9 global_segment_id=17775'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0100/4x4_Knee_Strikes_clip_35_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 175/236 bucket=boundary_9 global_segment_id=5559'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/001527_chunk_0002.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 176/236 bucket=boundary_9 global_segment_id=18940'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0048/Step_Out_Knee_Tuck_Punches_R_Clip1_clip_22_chunk_0001.npz --start_frame 150 --end_frame_exclusive 200 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 177/236 bucket=middle global_segment_id=17764'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/010928_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 178/236 bucket=middle global_segment_id=11647'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/000981_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 179/236 bucket=middle global_segment_id=5658'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Piloting_and_standing_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 180/236 bucket=middle global_segment_id=11783'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/013771_chunk_0002.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 181/236 bucket=middle global_segment_id=8182'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Simultaneously_Concession_and_walking_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 182/236 bucket=middle global_segment_id=6569'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0085/Side_Step_Shoulder_Press_Tociap_Undfr_clip_18_chunk_0000.npz --start_frame 150 --end_frame_exclusive 155 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 183/236 bucket=middle global_segment_id=160'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0027/Projectile_Schie_clip_76_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 184/236 bucket=middle global_segment_id=900'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/011596_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 185/236 bucket=middle global_segment_id=17641'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Sax_20_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 186/236 bucket=middle global_segment_id=14600'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/004697_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 187/236 bucket=middle global_segment_id=923'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Simultaneously_walking_and_Riding_the_Ferris_Wheel_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 188/236 bucket=middle global_segment_id=5154'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/animation/Ways_to_Pick_Up_a_Dollar_Dramatic_clip2_chunk_0001.npz --start_frame 100 --end_frame_exclusive 117 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 189/236 bucket=middle global_segment_id=5353'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0034/Other_I_clip_14_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 190/236 bucket=middle global_segment_id=20104'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Washing_things_and_walking_at_the_same_time_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 191/236 bucket=middle global_segment_id=6378'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/haa500/football_throw_5_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 192/236 bucket=middle global_segment_id=10580'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/standing_during_Riding_a_Carriage_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 193/236 bucket=middle global_segment_id=2680'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0038/Warrior_Flow_R_clip_10_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 194/236 bucket=middle global_segment_id=16265'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0062/Leg_Lift_To_Hip_Opener_R_clip_3_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 195/236 bucket=middle global_segment_id=8854'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humman/Opening_and_closing_step_1_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 196/236 bucket=middle global_segment_id=19830'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_Malimba_xylophone_8_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 197/236 bucket=quality_cross_high_pass global_segment_id=75'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0053.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 198/236 bucket=quality_cross_high_pass global_segment_id=78'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0053.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 199/236 bucket=quality_cross_high_pass global_segment_id=111'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0039/Arm_Circle_To_Butt_Kick_clip_32_chunk_0002.npz --start_frame 100 --end_frame_exclusive 149 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 200/236 bucket=quality_cross_high_pass global_segment_id=112'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0034/Reach_Up_To_Cross_Toe_Toucf_clip_2_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 201/236 bucket=quality_cross_high_pass global_segment_id=122'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/006322_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 202/236 bucket=quality_cross_high_pass global_segment_id=125'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0009/Fighting_Combo_Bombardierung_clip_2_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 203/236 bucket=quality_cross_high_pass global_segment_id=126'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0009/Fighting_Combo_Bombardierung_clip_2_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 204/236 bucket=quality_cross_high_pass global_segment_id=159'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0027/Projectile_Schie_clip_76_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 205/236 bucket=quality_cross_high_pass global_segment_id=172'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/game_motion/subset_0021/Long_Weapon_Splash_clip_26_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 206/236 bucket=quality_cross_high_pass global_segment_id=306'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/aist/subset_0008/Dance_Middle_Hip-hop_clip_23_chunk_0002.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 207/236 bucket=quality_cross_high_pass global_segment_id=334'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/013832_chunk_0002.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 208/236 bucket=quality_cross_high_pass global_segment_id=335'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/013832_chunk_0002.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 209/236 bucket=quality_cross_high_pass global_segment_id=405'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/jumps1/subject2_chunk_0040.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 210/236 bucket=quality_cross_high_pass global_segment_id=485'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/004585_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 211/236 bucket=quality_cross_high_pass global_segment_id=502'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/009351_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 212/236 bucket=quality_cross_high_pass global_segment_id=503'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/009351_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 213/236 bucket=quality_cross_high_pass global_segment_id=553'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/009294_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 214/236 bucket=quality_cross_high_pass global_segment_id=576'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0039/2_Squats_2_Jacks_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 215/236 bucket=quality_cross_high_pass global_segment_id=578'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0039/2_Squats_2_Jacks_chunk_0001.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 216/236 bucket=quality_cross_high_pass global_segment_id=579'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0039/2_Squats_2_Jacks_chunk_0001.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 217/236 bucket=quality_cross_high_reject global_segment_id=100'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/custom/squat_chunk_0019.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 218/236 bucket=quality_cross_high_reject global_segment_id=714'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/perform/step_out_sauna_1_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 219/236 bucket=quality_cross_high_reject global_segment_id=1063'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/004233_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 220/236 bucket=quality_cross_high_reject global_segment_id=1097'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/kungfu/Shaolin_Kung_Fu_Wushu_Basic_Tiger_Sword_Training_1_clip3_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 221/236 bucket=quality_cross_high_reject global_segment_id=1182'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/007104_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 222/236 bucket=quality_cross_high_reject global_segment_id=1457'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/sprint1/subject2_chunk_0028.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 223/236 bucket=quality_cross_high_reject global_segment_id=1829'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Lifting_things_while_standing_clip1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 224/236 bucket=quality_cross_high_reject global_segment_id=1963'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/humanml/006198_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 225/236 bucket=quality_cross_high_reject global_segment_id=2803'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_while_Putting_it_down_clip1_chunk_0000.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 226/236 bucket=quality_cross_high_reject global_segment_id=3482'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/fitness/subset_0098/Standing_Split_Pulse_To_Sout_R_clip_1_chunk_0000.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 227/236 bucket=quality_cross_high_reject global_segment_id=3496'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run2/subject4_chunk_0032.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 228/236 bucket=quality_cross_high_reject global_segment_id=3933'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/beat_drums_and_gongs_32_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 229/236 bucket=quality_cross_high_reject global_segment_id=4347'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/music/Play_the_violin_11_clip3_chunk_0002.npz --start_frame 150 --end_frame_exclusive 199 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 230/236 bucket=quality_cross_high_reject global_segment_id=4451'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/LAFAN1/run1/subject2_chunk_0018.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 231/236 bucket=quality_cross_high_reject global_segment_id=5347'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Swinging_during_standing_1_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 232/236 bucket=quality_cross_high_reject global_segment_id=5349'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/Swinging_during_standing_1_clip1_chunk_0001.npz --start_frame 100 --end_frame_exclusive 127 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 233/236 bucket=quality_cross_high_reject global_segment_id=5836'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/haa500/beer_pong_throw_3_clip1_chunk_0000.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 234/236 bucket=quality_cross_high_reject global_segment_id=6360'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_while_Grasping_Objects_clip1_chunk_0000.npz --start_frame 100 --end_frame_exclusive 150 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 235/236 bucket=quality_cross_high_reject global_segment_id=6442'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_during_Digging_1_clip1_chunk_0001.npz --start_frame 0 --end_frame_exclusive 50 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

echo '[module2_difficulty] 236/236 bucket=quality_cross_high_reject global_segment_id=6443'
env -u PYTHONPATH -u LD_LIBRARY_PATH python scripts/replay_npz.py --motion_file /home/l/whole_body_tracking_new/PHUMA_wbt_motions/g1_all/idea400/walking_during_Digging_1_clip1_chunk_0001.npz --start_frame 50 --end_frame_exclusive 100 --device cuda:0 --max_steps 150 --progress_interval 50
read -r -p '记录 notes 后按 Enter 继续...' _

