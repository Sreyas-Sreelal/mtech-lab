import socket
conn = socket.create_connection(("",8080))
conn.send(input("Enter command: ").encode())