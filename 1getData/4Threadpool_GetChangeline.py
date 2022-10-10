from bs4 import BeautifulSoup
from request_urls import request_url
import request_files
import os,datetime,logging
from concurrent.futures import ThreadPoolExecutor,wait
logger = logging.getLogger("mylogger")

#功能：根据筛选过后的api，进行源码的changeline的爬取
#修改参数:select的api的文件路径filename，以及new和old的路径名称

def useTheadPool(filename, func, batchnum_lastrun, threadcount,batchsize):

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
                        commit_url=line
                        future_task = pool.submit(func, commit_url)
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

def getChange(url):
    api_js=request_url(url)
    if api_js!= None:
        html=api_js["html_url"]
        print(html)
        resp = request_files.request_url(html)
        if resp==None:
            print(url)
        else:
            soup_html = BeautifulSoup(resp.text, 'lxml')
            # print(soup_html)
            files = soup_html.find("div", id="files")  # 找到id=files标签
            soup_files = BeautifulSoup(str(files), 'lxml')
            file_count = len(api_js["files"])
            # 其实可以不用要for 但是后面要用到file_sha,我暂时api打不开，分析不了
            for i in range(file_count):
                file_sha = api_js["files"][i]['sha']  # file_sha作为每一个文件名的唯一标识
                filename = api_js["files"][i]['filename']
                #修改筛选文件后缀
                if filename.endswith(".c") or filename.endswith(".cpp"):
                    try:
                        diff_files = soup_files.find("div", class_="js-file-content Details-content--hidden")
                        soup_diff = BeautifulSoup(str(diff_files), 'lxml')
                        # print(soup_diff)
                        if diff_files != None:
                            list_new = []
                            list_old = []
                            for tr in soup_diff.find_all('tr'):
                                soup_tr = BeautifulSoup(str(tr), 'lxml')
                                # print(soup_tr)
                                for td in soup_tr.find_all("td",
                                                           class_="blob-num blob-num-addition js-linkable-line-number"):
                                    line_number = td["data-line-number"]
                                    list_new.append(line_number)
                                for td in soup_tr.find_all("td",
                                                           class_="blob-num blob-num-deletion js-linkable-line-number"):
                                    line_number = td["data-line-number"]
                                    list_old.append(line_number)
                            line_str_new = " ".join(list_new)
                            line_str_old = " ".join(list_old)
                            # print(line_str)
                            # 某些文件删减行为0，只有添加行
                            if len(list_new) != 0:
                                # print(url, file_sha, line_str_new)
                                #修改
                                with open(r"improperAuthentication/improperAuthentication-new.txt", 'a')as f:
                                    f.write(file_sha + " " + line_str_new + "\n")
                            if len(list_old) != 0:
                                # print(url, file_sha, line_str_old)
                                #修改
                                with open(r"improperAuthentication/improperAuthentication-old.txt", 'a')as f:
                                    f.write(file_sha + " " + line_str_old + "\n")
                        else:
                            print("this file is not shown ！！！")

                    except Exception as e:
                        # pass
                        print(str(e))

if __name__=="__main__":
    # api="https://api.github.com/repos/alexo/wro4j/commits/c0ec4f6d7ce1130e92f0b67a5f74c41cae201645"
    # getChange(api)
    # os.chdir(r"E:\Pycharm\Python_Pycharm\getChange\javafiles")
    filename=r"data/improperAuthentication/improperAuthentication-selectapi.txt"

    logger = logging.getLogger("mylogger")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("changeline_" + filename.split('/')[1] + "_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    batchsize=20
    lastrun=6 #更改
    threadcount=20
    useTheadPool(filename=filename,func=getChange,batchnum_lastrun=lastrun,threadcount=threadcount,batchsize=batchsize)


