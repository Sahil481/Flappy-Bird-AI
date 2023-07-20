import train
import model
import os
import sys

def path_to_config() -> str:
    local_dir = os.path.dirname(__file__)
    return os.path.join(local_dir, 'config.txt')

def main():
    n = len(sys.argv)
    if n == 1:
        print("Syntax: python main.py [train/model]")
        return
    
    value = 0
    if "--limit" in sys.argv:
        try:
            value = int(sys.argv[sys.argv.index("--limit") + 1])
        except:
            print("Syntax: python main.py [train/model] [--limit] [value]")
            return
    else:
        value = 200

    if sys.argv[1] == "train":
        train.run(path_to_config(), value)
    elif sys.argv[1] == "model":
        model.run(path_to_config(), value)
    else:
        print("Syntax: python main.py [train/model]")
        return
    
if __name__ == "__main__":
    main()