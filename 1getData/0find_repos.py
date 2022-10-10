from request_urls import request_url
from calendar import monthrange

#功能:在github上爬取需要类型的仓库
#修改参数：修改year进行每一年仓库的爬取，两个api（爬取条件），以及保存文件名字

year =2012
months = ['01','02','03','04','05','06','07','08','09','10','11','12']
# time1 = ['T00:00:00Z','T01:00:00Z','T02:00:00Z','T03:00:00Z','T04:00:00Z','T05:00:00Z','T06:00:00Z','T07:00:00Z','T08:00:00Z','T09:00:00Z','T10:00:00Z','T11:00:00Z','T12:00:00Z',
#       'T13:00:00Z', 'T14:00:00Z', 'T15:00:00Z', 'T16:00:00Z', 'T17:00:00Z', 'T18:00:00Z', 'T19:00:00Z', 'T20:00:00Z','T21:00:00Z', 'T22:00:00Z', 'T23:00:00Z']
# time2 = ['T01:00:00Z','T02:00:00Z','T03:00:00Z','T04:00:00Z','T05:00:00Z','T06:00:00Z','T07:00:00Z','T08:00:00Z','T09:00:00Z','T10:00:00Z','T11:00:00Z','T12:00:00Z',
#       'T13:00:00Z', 'T14:00:00Z', 'T15:00:00Z', 'T16:00:00Z', 'T17:00:00Z', 'T18:00:00Z', 'T19:00:00Z', 'T20:00:00Z','T21:00:00Z', 'T22:00:00Z', 'T23:00:00Z', 'T24:00:00Z']
x = 1
for month in months:
    days = monthrange(year,x)[1]
    # days = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16',
    #         '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31']
    for day in range(1,days+1,1):
        if day<=9:
            day = '0'+str(day)
        # for t1, t2 in zip(time1, time2):
            #2016-03-21T14：11：00Z..2016-04-07T20：45：00Z
        # date = str(year) + '-' + month + '-' + str(day) + t1 + '..' + str(year) + '-' + month + '-' + str(day) + t2
        date = str(year) + '-' + month + '-' + str(day)
        #修改api以控制爬取条件
        api = "https://api.github.com/search/repositories?q=language:c+created:" + date
        total_count = request_url(api)['total_count']
        max_page = int(total_count / 100) + 2
        print(max_page)
        if max_page > 11:
            max_page = 11
        for page in range(1, max_page):
            # 修改api以控制爬取条件：按天爬取，原因在于github的api每次搜索结果最多返回1000条数据
            api = "https://api.github.com/search/repositories?q=language:c+created:" + date + "&per_page=100&page=" + str(
                page)
            api_json = request_url(api)
            repo_count = len(api_json['items'])
            print(repo_count)
            for i in range(repo_count):
                repo_name = api_json['items'][i]['full_name']
                print(repo_name, date)
                #修改年份文件名称
                with open(r'2012.txt', 'a')as f:
                    f.write(repo_name + '\n')
    x=x+1
