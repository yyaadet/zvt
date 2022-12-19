import socket


def open_connecton(host, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(5)
    try:
        sock.connect((host, port))
    except Exception as e:
        sock.close()
        raise e
    return sock


def binary_search(arr, ix):
    if len(arr) == 0:
        return False
    if ix == len(arr) - 1:
        return True
    if arr[ix] == arr[ix + 1]:
        return binary_search(arr, ix + 1)
    else:
        return False