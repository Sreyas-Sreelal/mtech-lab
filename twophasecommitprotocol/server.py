import socket
import threading
connections = []
stop = False


def accept_connections(server):
    global connections, stop
    while not stop:
        conn = server.accept()
        if conn:
            # conn[0].setblocking(0)
            connections.append(conn[0])
            print("total connected", len(connections))
            threading.Timer(0, recievefromclient, [conn[0]]).start()


def recievefromclient(conn):
    global connections, stop
    while not stop:
        try:
            msg = conn.recv(6)
            if msg == b'':
                connections.remove(conn)
                break
            elif msg == b'ABORT':
                print("Aborting committ")
        except Exception as e:
            print(str(e))
            connections.remove(conn)
            break


def start_server():
    global connections, stop
    server = socket.create_server(("", 8080))
    server.listen()
    server.setblocking(5)
    threading.Timer(0, accept_connections, [server]).start()
    try:
        while not stop:
            continue
    except KeyboardInterrupt:
        print("closing")
        stop = True
        exit(0)


start_server()
