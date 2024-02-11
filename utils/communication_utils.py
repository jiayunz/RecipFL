import pickle
import socket
import time
from tqdm import tqdm
import math

def recv(soc, buffer_size=1024, recv_timeout=10):
    # get message length
    try:
        soc.settimeout(recv_timeout)
        msg = soc.recv(buffer_size)
        msg = pickle.loads(msg)
        if msg['subject'] == 'header':
            data_len = msg['data']
            soc.sendall(pickle.dumps({"subject": "header", "data": "ready"}))
        else:
            raise Exception('Does not receive message header.')
            return None, 0
    except socket.timeout:
        print(f"A socket.timeout exception occurred after {recv_timeout} seconds. There may be an error or the model may be trained successfully.")
        return None, 0
    except BaseException as e:
        print(f"An error occurred while receiving header {e}.")
        return None, 0

    received_data = b""
    #recv_rounds = math.ceil(data_len / buffer_size)
    #for _ in tqdm(range(recv_rounds), total=recv_rounds):
    #    print(len(received_data), '/', data_len)
    while len(received_data) < data_len:
        #print(len(received_data), '/', data_len, 'buffer_size', buffer_size)
        try:
            soc.settimeout(recv_timeout)
            msg = soc.recv(buffer_size)
            if msg == b"":
                break
            received_data += msg
        except socket.timeout:
            print(
                f"A socket.timeout exception occurred after {recv_timeout} seconds. There may be an error or the model may be trained successfully.")
            return None, 0
        except BaseException as e:
            print(f"An error occurred while receiving data {e}.")
            return None, 0

    try:
        received_data = pickle.loads(received_data)
    except BaseException as e:
        print(f"Error Decoding the Client's Data: {e}.")
        return None, 0

    return received_data, 1


def send(soc, msg, buffer_size=1024):
    # msg: data bytes
    soc.sendall(pickle.dumps({"subject": "header", "data": len(msg)}))
    soc.recv(buffer_size)
    soc.sendall(msg)
