
import time
import zmq
import argparse
import numpy as np
import h5py as h5
import hdf5plugin

def parse_arguments():
  parser = argparse.ArgumentParser(
          description='Data Acquisition Process Simulator')

  parser.add_argument('--mode', type=int, required=True,
                      help='Data acqusition mod (0=detector; 1=simulate; 2=test)')

  parser.add_argument("--image_pv", help="EPICS image PV name.")
	
  parser.add_argument('--publisher_addr', default="tcp://*:50000",
                      help='Publisher addresss of data source process.')
  parser.add_argument('--publisher_hwm', type=int, default=0,
                      help='Sets high water mark value for publisher.')

  parser.add_argument('--synch_addr', help='Waits for all subscribers to join.')
  parser.add_argument('--synch_count', type=int, default=1,
                      help='Number of expected subscribers.')

  parser.add_argument('--simulation_file', help='File name for mock data acquisition. ')
  parser.add_argument('--d_iteration', type=int, default=1,
                      help='Number of iteration on simulated data.')
  parser.add_argument('--iteration_sleep', type=float, default=0,
                      help='Delay data publishing for each iteration.')

  return parser.parse_args()

def synchronize_subs(context, subscriber_count, bind_address_rep):
  # Prepare synch. sockets
  sync_socket = context.socket(zmq.REP)
  sync_socket.bind(bind_address_rep)
  
  counter = 0
  print("Waiting {} subscriber(s) to synchronize...".format(subscriber_count))
  while counter < subscriber_count:
    msg = sync_socket.recv() # wait for subscriber
    sync_socket.send(b'') # reply ack
    counter += 1
    print("Subscriber joined: {}/{}".format(counter, subscriber_count))


def test_daq(publisher_socket,num_images,slp=1):
  # creates a random nd array for transmitting 
  print("Creating random image data")
  dims = (num_images, num_images)
  image = np.array(np.random.randint(2, size=dims), dtype='uint16')
  for imageId in range(num_images):
    publisher_socket.send(image)
    time.sleep(slp)

  return imageId
   
def simulate_daq_file(publisher_socket, 
              input_f, iteration=1,
              slp=0): 
  print("Reading the Ptychography data from file")
  #t0=time.time()
  
  
  data = np.load(input_f)
  print(data.shape)
  
  data = np.array(data, dtype=np.dtype('float32'))

  tot_transfer_size=0
  start_index=0
  time0 = time.time()
  for it in range(iteration): # Simulate data acquisition
    print("Current iteration over dataset: {}/{}".format(it+1, iteration))
    for dchunk in data:
      publisher_socket.send(dchunk, copy=False)
      tot_transfer_size+=len(dchunk)
    time.sleep(slp)
  time1 = time.time()

  elapsed_time = time1-time0
  tot_kiBs = (tot_transfer_size*1.)/2**10
  nproj = iteration*len(data)
  print("Sent number of projections: {}; Total size (kiB): {:.2f}; Elapsed time (s): {:.2f}".format(nproj, tot_kiBs, elapsed_time))
  print("Rate (kiB/s): {:.2f}; (msg/s): {:.2f}".format(tot_kiBs/elapsed_time, nproj/elapsed_time))

  return iteration


def main():
  args = parse_arguments()

  # Setup zmq context
  context = zmq.Context()

# Publisher setup
  publisher_socket = context.socket(zmq.PUB)
  publisher_socket.set_hwm(args.publisher_hwm)
  publisher_socket.bind(args.publisher_addr)

  # 1. Synchronize/handshake with remote
  if args.synch_addr is not None:
    synchronize_subs(context, args.synch_count, args.synch_addr)

# 2. Transfer data
  time0 = time.time()
  if args.mode == 0: # Read data from PV
    print("need to do") #Infinite loop
  elif args.mode == 1: # Simulate data acquisition with a file
    print("Simulating data acquisition on file:")
    simulate_daq_file(publisher_socket=publisher_socket, 
              input_f=args.simulation_file, iteration=args.d_iteration,
              slp=args.iteration_sleep)
  elif args.mode == 2: # Test data acquisition
    test_daq(publisher_socket=publisher_socket, num_images=512,
              slp=args.iteration_sleep)
  else:
    print("Unknown mode: {}".format(args.mode));

  publisher_socket.send("end_data".encode())
  time1 = time.time()
  print("Total time (s): {:.2f}".format(time1-time0))


if __name__ == '__main__':
    main()
