import numpy as np
import struct
import socket
import psutil

M_SIZE = 1024
class UdpClient:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.pack_format = '<f'
        self.connect()
        
    def connect(self):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print('acceptting request .......')
        
    def get_sensor_data(self):
        memory = psutil.virtual_memory()
        self.data_ = memory.percent * np.random.random_sample()
        
    def send_(self):
        while True:
            self.get_sensor_data()
            send_len = self.client_socket.sendto(struct.pack(self.pack_format, self.data_), (self.host, self.port))
            rx_meesage, addr = self.client_socket.recvfrom(M_SIZE)
            print(f"[Server]: {rx_meesage.decode(encoding='utf-8')}")
        self.client_socket.close()




class UdpServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.pack_format = '<f'
        self.accept_socket()

    def accept_socket(self):
        self.server_sock = socket.socket(socket.AF_INET, type=socket.SOCK_DGRAM)
        self.server_sock.bind((self.host, self.port))
       
    def _receive(self):
        message, cli_addr = self.server_sock.recvfrom(M_SIZE)
        message = struct.unpack(self.pack_format, message)
        print('Send response to Client')
        self.server_sock.sendto('Success to receive message'.encode(encoding='utf-8'), cli_addr)
        return list(message)[0]

