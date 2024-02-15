import os
import shutil
import  time
if __name__ == '__main__':
    #1. build package
    print("*"*10,"Step 1","*"*10)
    time.sleep(2)
    os.system("cd build && python setup.py bdist_wheel")


    # install
    print("*"*10,"Step 2","*"*10)
    time.sleep(2)
    os.system("cd build && cd dist && pip install mtest-1.0-py3-none-any.whl")
    shutil.rmtree("./build/build")
    shutil.rmtree("./build/dist")
    shutil.rmtree("./build/mtest.egg-info")