import os
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model.")
    parser.add_argument("--port", default=6006,type=int)
    args = parser.parse_args()

    os.system('streamlit run WebServer/Server_SRD.py --server.port {}'.format(args.port))