import logging

import zmq

from nanotune.drivers.dac_interface import DACInterface

# from qcodes import Station

PORT = 7780


# to be used as:
# def main():

#     # initialise a connection to the instrument
#     scfg = Station('system1.yaml')
#     mips = scfg.load_instrument('dac')


#     # initialize the server
#     context = zmq.Context()
#     socket = context.socket(zmq.REP)
#     socket.bind(f"tcp://127.0.0.1:{PORT}")

#     print(f"Started server on localhost port {PORT}")

#     keep_going = True
#     while keep_going:
#         data = socket.recv()
#         # print(f'Got a message: {data}')
#         if data == b"kill":
#             socket.send(b"Received kill order, goodbye")
#             socket.close()
#             context.term()
#             keep_going = False
#         else:
#             reply = message_parser(mips, data.decode('utf-8'))
#             socket.send(reply.encode('utf-8'))


# if __name__ == "__main__":
#     print("Running main")
#     main()


def message_parser(mips: DACInterface, message: str) -> str:
    """
    A very crude proof-of-principle way of making the mips do something
    based on a message. The idea is that we send 'parametername;value' for set
    or 'parametername;?' for a query. ONLY works for float values.

    Returns:

    """
    if message.count(";") != 2:
        return "Illegal command"

    ch_id, parameter, value = message.split(";")
    channelid = int(ch_id) - 1  # channel list is zero indexed
    if channelid not in range(0, 64):
        return "No such parameter"

    if value == "?":
        try:
            # Get properties of qcodes parameter dac.chX.voltage
            if parameter in [
                "post_delay",
                "inter_delay",
                "step",
                "label",
                "name",
                "short_name",
            ]:
                val = getattr(mips.channels[channelid].voltage, parameter)
            # Get qcodes parameters
            else:
                val = getattr(mips.channels[channelid], parameter).get()

            return str(val)
        except Exception as e:
            logging.error(e)
            logging.error(f"Could not get dac.ch{channelid}.{parameter}")
            return f"Could not get dac.ch{channelid}.{parameter}"
    else:
        try:
            # Set properties of qcodes parameter dac.chX.voltage
            if parameter in [
                "post_delay",
                "inter_delay",
                "step",
                "label",
                "name",
                "short_name",
            ]:
                if parameter == "step" and value == "None":
                    value = "0"
                # if parameter not in ['label', 'name', 'short_name']:
                #     value = float(value)
                setattr(mips.channels[channelid].voltage, parameter, float(value))

            # Set qcodes parameters
            else:
                # if parameter == 'voltage':
                #     value = float(value)
                getattr(mips.channels[channelid], parameter).set(float(value))

            return f"Set dac.ch{channelid}.{parameter} to {value}"
        except Exception as e:
            logging.error(e)
            logging.error(f"Could not set dac.ch{channelid}.{parameter} to {value}")
            return f"Could not set dac.ch{channelid}.{parameter} to {value}"


class DACClient:

    PORT = 7780  # make sure that this agrees with the server

    def __init__(self) -> None:

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        # self.socket.connect(f"tcp://127.0.0.1:{self.PORT}")
        self.socket.connect(f"tcp://127.0.0.1:{self.PORT}")

    def send(self, message: str) -> None:
        """
        Send a message to the server
        """
        self.socket.send(message.encode("utf-8"))
        return self.socket.recv().decode("utf-8")  # recieve OK
