import socket
import pickle

"""#length of the message length
HEADERSIZE = 10
SERVER_IP = ''
SERVER_PORT = 50008"""

class TCPClient:
    def __init__(self, header_size, server_ip, server_port):
        self.header_size = header_size
        self.server_ip = server_ip
        self.server_port = server_port
        self.socket = None

    def connect2server(self):
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.server_ip, self.server_port))

            server_name = str(self.server_ip)
            if self.server_ip == '':
                server_name = 'localhost'

            print("Connected to the server: " + server_name + ":" + str(self.server_port))
            return True
        except Exception as e:
            self.socket.close()
            print("Cannot connect to the server! " + str(e))
            return False

    def send2server(self, obj2send):
        if not self.connect2server():
            return False
        try:
            msg = pickle.dumps(obj2send)
            msg = bytes(f"{len(msg):<{self.header_size}}", 'utf-8')+msg
            self.socket.sendall(msg)
            self.socket.close()
            print("Message sent!")
            return True
        except Exception as e:
            self.socket.close()
            print("Cannot send message to the server! " + str(e))
            return False
