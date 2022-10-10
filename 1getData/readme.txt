getData是利用github的api对github的数据进行爬取
参考github api官方文档=>https://developer.github.com/v3/

大体思路是：
1.先爬取github所需数据的仓库（比如爬取c语言仓库）（0find_repos.py）
2.再通过获取到的仓库，爬取所需数据的commits（比如爬取CWE-074的数据）,以json文件的形式保存(1get_commits.py)
3.把所有的commits中的api提取出来(2get_allapi.py)
4.在所有api中筛选出每次提交只修改了一个文件的commits(3get_selectapi.py)
5.利用筛选过后的api下数据的源码和changeline（4Theadpool_Get_Codefiles.py和4Threadpool_GetChangeline.py）
最后会通过changeline对源码进行切片与分类