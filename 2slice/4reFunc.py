import os
import re
import shutil

def judge(func_name, para_name):
    Flag = False
    #authentication
    # func_names = ['login', 'auth', 'session', 'check', 'control', 'match', 'validate', 'grant', 'token']
    # para_names = ['userid', 'session', 'auth', 'token', 'username']
    #changePassage
    # func_names = ['password', 'pw', 'passwd', 'auth', 'authpasswd','verify', 'change','client','user']
    # para_names = ['user', 'password', 'pw', 'passwd']
    #privilegeMangement
    # func_names = ['config', 'wap', 'alloc', 'request', 'auth', 'management', 'privacy', 'user']
    # para_names = ['config', 'alloc', 'auth', 'user']
    #accountInformation
    # func_names = ['account', 'message', 'alloc', 'history', 'msg','user','information']
    # para_names = ['addr', 'auth', 'user', 'message', 'msg']
    #pm
    # func_names = ['account','history', 'user', 'information']
    # para_names = ['addr', 'user']

    #项目cjson
    # func_names = ['get_object_item', 'cjson_get_object_item_case_sensitive_should_get_object_items']
    # para_names = []
    #项目 linux
    # func_names = ['uvesafb_setcmap', 'cdrom_ioctl_media_changed','kill_something_info','futex_requeue','__get_data_block']
    # para_names = []
    #sqllit
    func_names = ['sqlite3VXPrintf', 'sqlite3VdbeExec']
    para_names = []

    # for f_name in func_names:
    #     if f_name in func_name.lower():
    #         Flag = True
    # for para in para_names:
    #     if para in para_name.lower():
    #         Flag = True

    for f_name in func_names:
        if f_name in func_name:
            Flag = True
    for para in para_names:
        if para in para_name:
            Flag = True

    return Flag

#通过函数名参数名进行筛选
def refineRe(rootdir,goodPath,badPath):
    # pattern = re.compile(r"(?:(?:static)\s)*(\w+)\s?(.*)\s*(?:(\(.*\))(\s*{))",re.I)
    #sard github
    pattern = re.compile(r"(?:(?:static)\s)*(\w+)\s*(.*)\s?(?:([(].*\s*.*[)]).*(\s*{))", re.I)
    #项目
    # pattern = re.compile(r"(?:(?:static)\s)*(\w+)?\s*(.*)\s*(?:([(]\s*.*\s*.*\s*[)]).*(\s*{))", re.I)

    filenames = os.listdir(rootdir)

    if not os.path.exists(badPath):
        os.makedirs(badPath)
    if not os.path.exists(goodPath):
        os.makedirs(goodPath)

    count = 0
    for filename in filenames:
        file = os.path.join(rootdir, filename)
        with open(file, 'r', encoding='utf-8')as f:
            text = f.read()
            result = re.findall(pattern, text)
            if not result:
                count = count + 1
                print(file)
            else:
                func = result[0]
                func_name = func[1]
                para_name = func[2]
                Flag = judge(func_name, para_name)

                if Flag:
                    print(func)
                    shutil.copy(file, badPath)

                else:
                    # print(func)
                    shutil.copy(file, goodPath)
    print(count)


def judgeF(func_name):
    Flag = False
    #项目cjson
    # func_names = ['get_object_item', 'cjson_get_object_item_case_sensitive_should_get_object_items']
    #项目 linux
    # func_names = ['uvesafb_setcmap', 'cdrom_ioctl_media_changed','kill_something_info','futex_requeue','__get_data_block']
    #sqllit
    # func_names = ['sqlite3VXPrintf', 'sqlite3VdbeExec']
    #libav
    func_names = ['range_decode_culshift', 'vc1_decode_frame','ff_mpa_synth_filter_float','decode_frame','ff_vc1_parse_frame_header_adv',
                  'vc1_decode_i_block_adv','ff_vc1_pred_dc','vc1_decode_p_mb_intfi','in_table_init16','mov_probe']
    for f_name in func_names:
        if f_name in func_name:
            Flag = True
    return Flag

def refineFunc(rootdir,goodPath,badPath):
    # pattern = re.compile(r"(?:(?:static)\s)*(\w+)\s?(.*)\s*(?:(\(.*\))(\s*{))",re.I)
    #只匹配函数名
    pattern = re.compile(r"(?:(?:static)\s)*(\w+)?\s(.*)(?=\()", re.I)
    filenames = os.listdir(rootdir)

    if not os.path.exists(badPath):
        os.makedirs(badPath)
    if not os.path.exists(goodPath):
        os.makedirs(goodPath)

    count = 0
    for filename in filenames:
        file = os.path.join(rootdir, filename)
        with open(file, 'r', encoding='utf-8')as f:
            text = f.read()
            result = re.findall(pattern, text)
            if not result:
                count = count + 1
                print(file)
            else:
                func = result[0]
                func_name = func[1]
                Flag = judgeF(func_name)

                if Flag:
                    print(func)
                    shutil.copy(file, badPath)

                else:
                    # print(func)
                    shutil.copy(file, goodPath)
    print(count)

#从文本中进行筛选
def textRe(rootdir,goodPath,badPath):
    filenames = os.listdir(rootdir)

    if not os.path.exists(badPath):
        os.makedirs(badPath)
    if not os.path.exists(goodPath):
        os.makedirs(goodPath)

    func_names = ['password', 'pw', 'passwd']
    for filename in filenames:
        file = os.path.join(rootdir, filename)
        with open(file, 'r', encoding='utf-8')as f:
            text = f.read()
            for t in func_names:
                if t in text:
                    shutil.copy(file, badPath)
                    break
            else:
                shutil.copy(file, goodPath)

#测试单个文件
# file = r"5cd439c8-8620-46e7-b34e-3bda65d76968.c"
# # pattern = re.compile(r"(?:(?:static)\s)*(\w+)?\s*(.*)\s*(?:([(]\s*.*\s*.*\s*[)]).*(\s*{))", re.I)
# pattern = re.compile(r"(?:(?:static)\s)*(\w+)?\s(.*)(?=\()", re.I)
# # file = r"D:\XRZ\Data\joern\github\SensitiveOperations\0切片\changePassword\merge\0af779a4-1344-4d3d-872c-732ae8030bfc.c"
# with open (file, 'r', encoding='utf-8')as f:
#     text = f.read()
#     result = re.findall(pattern, text)
#     if not result:
#         print(file)
#     else:
#         func = result[0]
#         func_name = func[1]
#         para_name = func[2]
#         Flag = judge(func_name, para_name)

if __name__ == '__main__':
    # rootdir = r"D:\XRZ\Data\joern\github\SensitiveOperations\testData\1切片\accountInformation\slice"
    # goodPath = r"D:\XRZ\Data\joern\github\SensitiveOperations\testData\2筛选\ai1\good"
    # badPath = r"D:\XRZ\Data\joern\github\SensitiveOperations\testData\2筛选\ai1\bad"
    # refineRe(rootdir,goodPath,badPath)
    rootdir = r"C:\Users\WHT\Desktop\vul\slice\cjson\good"
    goodPath = r"D:\XRZ\Data\joern\project\libav\libav-vul\2classify\good"
    badPath = r"D:\XRZ\Data\joern\project\libav\libav-vul\2classify\bad"
    refineFunc(rootdir, goodPath, badPath)
