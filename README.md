# Inference-on-streamed-data


The ptychography data is read from the X_test_NCHW.npy and is send over the network through zeromq.
Inference code is added in the client_AGX_np_v1 python code. 

Commands that can be used for running the server and client. 

python3 server_detector_np_v1.py --mode=1 --publisher_addr="tcp://127.0.0.1:5555" --synch_addr="tcp://127.0.0.1:5556" --iteration_sleep=1 --simulation_file="X_test_NCHW.npy"

python3 client_AGX_np_v1.py --publisher_address="tcp://127.0.0.1:5555" --synchronize_subscriber --publisher_rep_address="tcp://127.0.0.1:5556"


Average Inference time ~ 7 ms using synchronous execution


**Note**: git-lfs required for accessing the test data
