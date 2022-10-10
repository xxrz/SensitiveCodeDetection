import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, wait
from request_urls import request_url
logger = logging.getLogger("mylogger")

#功能：在得到的所有api中筛选出每次提交只修改了一个文件的commits
#修改参数:修改所有api下的路径filename，以及最后保存的文件名

#并行爬取
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

#此处筛选每次提交只修改了一个文件的api
def select_api(commit_url):
    api_js = request_url(commit_url)
    if api_js is not None:
        if "files" in api_js.keys():
            if len(api_js["files"]) == 1:
                #记得修改写出来的文件名
                with open(r'data/improperAuthentication/improperAuthentication-selectapi.txt', 'a', encoding='utf-8') as f:
                    f.write(api_js['url'] + "\n")
                print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))



if __name__ == '__main__':
    filename="data/improperAuthentication/improperAuthentication-allapi.txt"
    batchsize=20
    threads_n =20
    lastrun=717#从哪里停止，从下一个继续run

    logger = logging.getLogger("mylogger")
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler("selectapi_" + filename.split('/')[1] + "_log.txt")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    useTheadPool(filename=filename, func=select_api, batchnum_lastrun=lastrun,
                     threadcount=threads_n,batchsize=batchsize)

    print("end!!")
