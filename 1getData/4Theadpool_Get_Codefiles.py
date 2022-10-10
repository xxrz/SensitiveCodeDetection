import datetime
import logging
import os
from concurrent.futures import ThreadPoolExecutor, wait
from request_urls import request_url
import request_files
logger = logging.getLogger("mylogger")

#功能：根据筛选过后的api，进行源码的爬取
#修改参数:select的api的文件路径filename，以及保存的文件名

def useTheadPool( filename, func, batchnum_lastrun, threadcount,batchsize,old_file,new_file):

    pool = ThreadPoolExecutor(threadcount)  # 不填则默认为cpu的个数*50
    future_tasks = []
    try:
        f = open(filename, 'r')
        finished = 0
        line_count = 0
        batchnum_now = 0
        old_finished = 0
        for line in f.readlines():
            if line.startswith("http"):
                line = line.strip()

                line_count = line_count + 1
                if not batchnum_now == int((line_count - 1) / batchsize + 1):
                    batchnum_now = int(line_count / batchsize + 1)
                    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  "  + " BATCH " + str(
                        batchnum_now) + " start")
                    logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + "  "  + " BATCH " + str(
                        batchnum_now) + " start")

                if batchnum_now < batchnum_lastrun:
                    continue
                else:

                    if len(future_tasks) < batchsize:
                        # kwargs = []
                        # kwargs.append(year)
                        # kwargs.append(line)
                        # print(kwargs)
                        commit_url=line
                        future_task = pool.submit(func, commit_url,old_file,new_file)
                        future_tasks.append(future_task)
                    if len(future_tasks) == batchsize:

                        for future_task in future_tasks:
                                if not future_task.result() == None:
                                        print(future_task.result())
                                if future_task.done():
                                    finished = finished + 1

                        wait(future_tasks)
                        print("finished，before " + str(line_count))
                        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " BATCH =========>" + str(
                            batchnum_now) + " end" + "   finished:" + str(finished) + "   threadcount: " + str(threadcount))
                        logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " BATCH =========>" + str(
                            batchnum_now) + " end" + "   finished:" + str(finished) + "   threadcount: " + str(threadcount))

                        if not finished == batchsize:
                            print("finised!=batchsize  something wrong! run this batch again: " + str(batchnum_now))
                            logger.info("finised!=batchsize  something wrong! run this batch again: " + str(batchnum_now))
                            exit(1)
                        else:
                            future_tasks = []
                            finished = 0
        f.close()

    except StopIteration as e:
        pass
    pool.shutdown(wait=True)

def download(commit_url,old_file,new_file):
    api_js = request_url(commit_url)
    if api_js != None:
        if len(api_js["parents"])==0:
             print("commit has no parent!")
        else:
                parent_hash = api_js["parents"][0]["sha"]
                files = api_js["files"]
                for i in range(len(files)):
                    if files[i]["filename"].endswith(".c"):
                        raw_url = files[i]["raw_url"]
                        raw_url_hash = raw_url.split("/")[6]  # raw_url的哈希值
                        sha_filename = raw_url.split("raw/")[1]
                        repo_name = raw_url.split("raw/")[0].split("https://github.com/")[1]
                        #拼接字符串，生成raw_url的连接
                        origin_raw_url="https://raw.githubusercontent.com/"+repo_name+sha_filename
                        parent_raw_url = origin_raw_url.replace(raw_url_hash, parent_hash)  # parent_raw_url的哈希值
                        raw_file_fileName = files[i]["sha"]  # 获取每一个文件名的sha，作为文件名唯一标识
                        # try:
                        resp_child = request_files.request_url(origin_raw_url)
                        resp_parent = request_files.request_url(parent_raw_url)
                        if resp_child!=None and resp_parent!=None:
                            with open(new_file + "/" + raw_file_fileName + ".txt", 'w',encoding='utf-8')as f:
                                f.write(resp_child.text)

                            with open(old_file + "/" + raw_file_fileName + ".txt", 'w',encoding='utf-8')as f:
                                f.write(resp_parent.text)
                            print("done",datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                        else:
                            print("NOT FOUND")
                                # print(origin_raw_url, parent_raw_url)
                        # except Exception as e:
                        #     print(str(e))


if __name__ == '__main__':
    #修改
    file = "improperAuthentication"
    # file = "cwe-119"
    logger = logging.getLogger("mylogger")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("log_"+file+"_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    new_file=os.getcwd()+"\\"+file+"\\new_files"
    old_file=os.getcwd()+"\\"+file+"\\old_files"
    if not os.path.exists(old_file):
        os.makedirs(old_file)
    if not os.path.exists(new_file):
        os.makedirs(new_file)

    # filename=file + "_sum.txt"
    # filename = "api_" + file + "_fromGithub.txt"
    #修改
    filename = r"data/improperAuthentication/improperAuthentication-selectapi.txt"
    batchsize=10
    threads_n =20
    # lastrun = 40
    lastrun=0#从哪里停止，从下一个继续run

    useTheadPool(filename=filename, func=download, batchnum_lastrun=lastrun,
                     threadcount=threads_n,batchsize=batchsize,old_file=old_file,new_file=new_file)

    print("end!!")
