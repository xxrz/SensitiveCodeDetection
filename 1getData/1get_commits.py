import csv
import datetime
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor,wait

from request_urls import request_url
import logging

#功能：爬取仓库中需要类型的commits,保存在json中
#修改参数：修改year，bugtype爬取不同漏洞类型不同年份的commits

finished=0
global_data = threading.local()

def readCSV(year):
    with open(year+".csv") as csvfile:
        csv_reader = csv.reader(csvfile) # 使用csv.reader读取csvfile中的文件
        header = next(csv_reader)  # 读取第一行每一列的标题
        line=next(csv_reader)
        while line:
            line=next(csv_reader)
            yield line[0]
        yield None

#获取commits函数
def visit( urls, line, year,line_count):
        global finished
        global_data.keyword=0
        for url in urls:
            global_data.keyword=global_data.keyword+1
            bugtype=url.split("+")[-1]
            r=request_url(url)
            #进行了修改
            if not "total_count" in r.keys():
                logger.info("%-40s keyword: %-20s don't have total_count" % (line, bugtype))
            else:
                logger.info("%-40s keyword: %-20s commit num:%d"%(line,bugtype,r["total_count"]))
                if r["total_count"]==0:
                    global_data.total_count=0
                    pass
                else:
                    global_data.total_count = r["total_count"]
                    reponame=line.replace("/","#")
                    with open(year+"/"+reponame+"$"+str(global_data.keyword)+".json","w",encoding="utf-8")as f:
                        json.dump(r,f)
        finished=finished+1

if __name__ == '__main__':
        # os.chdir(r"C:\Users\troye sivan\Documents\GitHub\collect_github_data")
    #     year = sys.argv[1]
    #     batchnum_lastrun = int(sys.argv[2])
    # for i in range(2011,2018):

        year=str(2019)  ##2019 479  2020 462
        batchsize = 50
        #有可能在爬取数据的过程中会卡，需要修改batchnum_lastrun更改至当前batch的下一个
        batchnum_lastrun = 479
        threadcount = 50

        #修改bugtype的字符串
        bugtype = ["authentication"]
        #account information
        logname = bugtype[0].replace(" ", "_")
        # logname="all_commits"
        logger = logging.getLogger("mylogger")
        logger.setLevel(level=logging.INFO)
        time_now = datetime.datetime.now().strftime('%Y-%m-%d')

        handler = logging.FileHandler("log_" + year + "_" + logname + ".txt")
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        t1 = time.time ()
        if not os.path.exists(year):
             os.makedirs (year)

        pool = ThreadPoolExecutor (threadcount)  # 不填则默认为cpu的个数*5
        start = time.time ()
        future_tasks=[]
        try:
            #读取收集仓库文件
            f=open ('cRepos/'+year+'.txt', 'r')
            line_count=0
            batchnum_now=0
            old_finished=0
            for line in f.readlines ():
                line=line.strip()
                line_count=line_count+1
                if not batchnum_now==int((line_count-1)/batchsize+1):

                     batchnum_now=int(line_count/batchsize+1)
                     print (datetime.datetime.now ().strftime ('%Y-%m-%d %H:%M:%S') + str(bugtype) + "  "+year+" BATCH " + str (batchnum_now) + " start")
                     logger.info(datetime.datetime.now ().strftime ('%Y-%m-%d %H:%M:%S') + "  "+year+" BATCH " + str (batchnum_now) + " start")

                if batchnum_now<batchnum_lastrun:
                    continue
                else:
                    #爬虫搜索条件
                    urls=["https://api.github.com/search/commits?q=repo:" + line +"+"+type+"&per_page=100" for type in bugtype]
                    if len(future_tasks)<batchsize:
                        future_task = pool.submit (visit,urls, line,year,line_count)
                        future_tasks.append(future_task)
                    if len(future_tasks)==batchsize:
                        for future_task in future_tasks:
                            if future_task.result()!=None:
                                   print(future_task.result())
                        wait (future_tasks)
                        print ("finished，before " + str (line_count))

                        live=finished-old_finished
                        old_finished=finished
                        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " BATCH =========>" + str(
                            batchnum_now) + " end" + "   finished:" + str(live) + "   threadcount: " + str(threadcount))
                        logger.info(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + " BATCH =========>" + str(
                            batchnum_now) + " end" + "   finished:" + str(live) + "   threadcount: " + str(threadcount))

                        if not live==batchsize:
                            print("finised!=batchsize  something wrong! run this batch again: "+str(batchnum_now))
                            logger.info("finised!=batchsize  something wrong! run this batch again: "+str(batchnum_now))
                            exit(1)
                        else:
                            future_tasks = []
            f.close()

        except StopIteration as e:
            pass
        pool.shutdown(wait=True)
        print(time.time() - t1)
        print("write to file")



