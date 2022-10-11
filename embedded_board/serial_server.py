#!/usr/bin/python3
import time
import serial

print("UART Demonstration Program")
print("Raspberry Pi4 Server")

def serial_server(server_port, path):
    # Wait a second to let the port initialize
    try:
        while True:
            if server_port.inWaiting() > 0:

                # serial readlines
                data = server_port.readline()

                with open(path, 'a') as f:
                    f.write(data)
                    
                # print(data)
                server_port.write(data)

                if data == "\r".encode():
                    # For Windows boxen on the other end
                    server_port.write("\n".encode())


    except KeyboardInterrupt:
        print("Exiting Program")

    except Exception as exception_error:
        print("Error occurred. Exiting Program")
        print("Error: " + str(exception_error))

    finally:
        server_port.close()
        pass

if __name__ == "__main__":
    server_port = serial.Serial(
    port="/dev/ttyS0",
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    )