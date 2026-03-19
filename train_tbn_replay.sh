# python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_6
# JSON에 weights_path가 있으면 그걸 사용, 없으면 커맨드라인에서 지정 가능
# python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_7 -c max_prob -g 0
python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_7 -c max_prob -g 0 -w weights/mmea_tbn_replay_auxiliary_head_v2_7_rgbgyroacce_ep50_bs8_pb1_fr0_inc4_mem320_train/Oct11_22-25-22
python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_7 -c entropy -g 0 -w weights/mmea_tbn_replay_auxiliary_head_v2_7_rgbgyroacce_ep50_bs8_pb1_fr0_inc4_mem320_train/Oct14_01-52-45
python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_7 -c energy -g 1 -w weights/mmea_tbn_replay_auxiliary_head_v2_7_rgbgyroacce_ep50_bs8_pb1_fr0_inc4_mem320_train/Oct14_01-52-50
python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_7 -c margin -g 1 -w weights/mmea_tbn_replay_auxiliary_head_v2_7_rgbgyroacce_ep50_bs8_pb1_fr0_inc4_mem320_train/Oct14_01-52-56
python main.py -d mmea -m tbn_replay -f auxiliary_head_v2_10 -c max_prob -g 1 -w /workspace/MMEA-OWCL/weights/mmea_tbn_replay_auxiliary_head_v2_10_rgbgyroacce_ep50_bs8_pb1_fr0_inc4_mem320_train/Oct14_19-26-59