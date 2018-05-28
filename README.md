# Click prediction on Criteo 1Tb dataset

##### By: Traci Lim

---

This repository holds a report in the form of a jupyter notebook, which focused on the prediction of the click theough rate of display ads, while managing the following objectives:

- Utilize Microsoft Azure HDInsight clusters for large-scale machine learning.
- Briefly summarize the key concepts behind algorithms used.
- Address the problem of class imbalance by resampling methods.
- Give a concise experimental analysis with variations in data and  number of worker nodes/partitions, for understanding the scalability and  distributed performance of algorithms in MLlib.
- Evaluate results and conclude findings.



The following parts of this markdown file highlights the key results from the report. 

---

### Experimental Setup

To configure the Spark application to use up as much of the cluster setup as possible, we ran a long series of tests on different configurations, and finalized the scores below. 

Cluster specifications table:

|         Cluster type         | Cores per Head Node | Cores per Worker Node | Memory per Node (Head, Worker) | Total Cores |    No. of Nodes    |
| :--------------------------: | :-----------------: | :-------------------: | :----------------------------: | :---------: | :----------------: |
| Spark 2.1 on Linux (HDI 3.6) |          4          |           8           |         (28GiB, 56GiB)         |     40      | 2x Head, 4x Worker |

Spark configuration options table:


| Master                                 | Yarn Cilent |
| -------------------------------------- | ----------- |
| `yarn.nodemanager.resource.memory-mb`  | 56320mb     |
| `yarn.nodemanager.resource.cpu-vcores` | 7           |
| `num-executors`                        | 6           |
| `executor-cores`                       | 4           |
| `executor-memory`                      | 11060mb     |
| `spark.dynamicAllocation.enabled`      | true        |
| `spark.shuffle.service.enabled`        | true        |
| `spark.driver.cores`                   | 4           |
| `spark.driver.memory`                  | 11060mb     |

---

### Methodology

![methodologyflowchart](https://raw.githubusercontent.com/longwind48/criteo_click_prediction_sparkml/master/methodologyflowchart.png)

---

### Conclusion

The cluster was set up with the kind support from **Microsoft Azure Sponsorship**, that generously provided 60 cores for the expense of this project. However, building machine learning pipelines on 1.3Tb of data requires many more cores to run efficiently and manageably. The figure below was taken from Dunner et al.[24], a paper that compared the performance of logistic regression on several machine learning libraries, on the Criteo dataset.

![logisticregcriteo1](https://raw.githubusercontent.com/longwind48/criteo_click_prediction_sparkml/master/logisticregcriteo.png)

In the chart above, Rambler Digital Solutions[25] deployed a Spark 2.1.0 with total 512 cores and 2TB of memory, and trained Logistic Regression on 4.2 billion training instances, taking approximately 100 minutes. Google[26] trained a Logistic Regression model using Tensorflow, and reported using 60 workers and 29 parameter machines for the same data, clocking a very competitive test logloss score and training time. Dünner's paper on Snap ML, a new machine learning library that supports GPU accelerators in a distributed environment, trained a Logistic Regression model using 16 GPUs, clocking an impressive score. This comparison is a clear indication of the competitiveness and innovation of the development in scalable machine learning.

------

With the 40 cores set up in this project, it was only possible to train on only a subset of the dataset. We still tried to emulate the existence of big data by working on millions of instances. The results of this project are summarised in the figure below.
![logisticregcriteo](https://raw.githubusercontent.com/longwind48/criteo_click_prediction_sparkml/master/results.png)

As expected, the number of worker nodes is held accountable for the running time of the creation of pipelines and model training, because more cores allows for a higher level of parallelism. Repartitioning time was constant throughout the benchmark test because repartitioning was executed with only one node, with zero parallelism. The accuracy and AUC scores between different data sizes were rather close, which could be explained by the similarities between datasets. Since they were sampled with the same seed, on purpose, we can observe how scalability in data affects results, which, in this case, it is clear that scalability did not yield a significant improvement in results. This is especially common when the nature of data does not 'explain' enough. When sampling, it can be easy to introduce bias by selecting a dataset that misrepresents or underrepresents the real cases, which will distort the results. To improve the quality of data, datasets should always be analyzed and preprocessed with the addition of newly engineered features that improve predictive power, as well as proper means of feature selection. The process of sampling should also be done with caution, drawing samples from multiple days could be one way to decrease the bias. 

There are many takeaways for this project, large scale machine learning tasks on Spark clusters requires knowledge not just methods to optimise predicitons, but also the Spark achitecture, YARN resource manager, and Spark tuning configurations. Cluster-mode in Spark requires sufficient comprehension to take advantage of the parallelism framework, which leads to proper tuning configurations to maximize the resources of the cluster. The Spark configurations used on this project was finalzed after a long series of testing, while validating the the efficiency of the memory allocation and parallelism of jobs between worker nodes. A handful of improvements could be made to this project, such as attempting a different set of classifiers to further explore the optimization of predicted scores, deploying more cores to better handle the scalability and parallelism, as well as attempting feature engineering and feature selection. 

The learning process of machine learning certainly feels like taking a drink from a firehose, however, it was very rewarding and worthwhile. On the whole, this opportunity has granted an interesting outlook on the domain of large scale machine learning. 

---

### References

[1]: Junxuan Chen, B. S.-S. (n.d.). [Deep CTR Prediction in Display Advertising]( http://wnzhang.net/share/rtb-papers/deep-ctr-display.pdfhttp://wnzhang.net/share/rtb-papers/deep-ctr-display.pdf). *ACM Multimedia Conference 2016.*
Hangzhou, China: Department of Computer Science and Engineering, Shanghai Jiao Tong University. 

[2]: [*Create an Apache Spark cluster in Azure HDInsight*.](https://docs.microsoft.com/en-gb/azure/hdinsight/spark/apache-spark-jupyter-spark-sql) (1 March, 2018). Retrieved from Microsoft.

[3]: [Deep Dive into Spark SQL Catalyst Optimizer](https://databricks.com/blog/2015/04/13/deep-dive-into-spark-sqls-catalyst-optimizer.html). (13 April, 2015) Retrieved from Databricks.

[4]: *[Project Tungsten: Bringing Apache Spark Closer to Bare Metal](https://databricks.com/blog/2015/04/28/project-tungsten-bringing-spark-closer-to-bare-metal.html)*. (28 April, 2015). Retrieved from Databricks.

[5]: *[Using Azure ML to Build Clickthrough Prediction Models](https://blogs.technet.microsoft.com/machinelearning/2015/11/03/using-azure-ml-to-build-clickthrough-prediction-models/)*. (3 November, 2015). Retrieved from Microsoft Machine Learning Blog.

[6]: Nandi, A. (2015). *Spark for Python Developers.* Packt Publishing.

[7]: Pentreath, N. (2017). *[Feature Hashing for Scalable Machine Learning](https://databricks.com/session/feature-hashing-for-scalable-machine-learning)*. Retrieved from Databricks.

[8]: Hui Zou, T. H. (2005). [Regularization and variable selection via the Elastic Net](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.124.4696). *Journal of the Royal Statistical Society, Series B*, 301-320.

[9]: *[Logistic Regression]( https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier)*. (n.d.). Retrieved from Apache Spark.

[10]: Fridley, J. (2010, February 22). [Tree models in R]( http://plantecology.syr.edu/fridley/bio793/cart.html ). Retrieved from Plantecology.

[11]: Marius-Constantin, V. ,.-P. (2009). *[Multilayer Perceptron and Neural Networks](http://www.wseas.us/e-library/transactions/circuits/2009/29-485.pdf) .* Romania: Faculty of Electromechanical and Environment Engineering, University of Craiova. 

[12]: *[Multilayer Perceptron Classifier](https://spark.apache.org/docs/latest/ml-classification-regression.html#multilayer-perceptron-classifier)*. (n.d.). Retrieved from Apache Spark.

[13]: Ulanov, A. (n.d.). *[A Scalable Implementation of Deep Learning on Spark](https://databricks.com/session/a-scalable-implementation-of-deep-learning-on-spark)*. Retrieved from Databricks.

[14]: Longadge, R., Dongre, S. S., & Malik, L. (2013, February). Class Imbalance Problem in Data Mining: Review. International Journal of Computer Science and Network 

[15]: Malouf, Robert (2002). [*A comparison of algorithms for maximum entropy parameter estimation*](https://web.archive.org/web/20131101205929/http://acl.ldc.upenn.edu/W/W02/W02-2018.pdf). Proc. Sixth Conf. on Natural Language Learning (CoNLL). pp. 49–55. Archived from [the original](http://acl.ldc.upenn.edu/W/W02/W02-2018.pdf) on 2013-11-01.

[16]: Wang, W. (9 June, 2016). *[PySpark tutorial – a case study using Random Forest on unbalanced dataset](https://weiminwang.blog/2016/06/09/pyspark-tutorial-building-a-random-forest-binary-classifier-on-unbalanced-dataset/)*. Retrieved from Weimin Wang Blog.

[17]: Milan Vojnovic, C. Y. (1 March, 2018). *[Logistic Regression Pipeline](https://github.com/lse-st446/lectures/blob/master/week07/class/logistic_regression_pipeline.ipynb)*. Retrieved from Github.

[18]: Nandi, A. (2015). *Spark for Python Developers.* Packt Publishing.

[19]: Karau, H. (2016). *High Performance Spark: Best Practices for Scaling and Optimizing Apache Spark.* O′Reilly.

[20]: Drabas, T. (2017). *Learning PySpark.* Packt Publishing.

[21]: [*How-to: Tune Your Apache Spark Jobs (Part 2)*.](https://blog.cloudera.com/blog/2015/03/how-to-tune-your-apache-spark-jobs-part-2/) (30 March, 2018). Retrieved from Databricks.

[22]: Gurbuzbalaban, M., Ozdaglar, A., Parrilo, P. A., & Vanli, N. (2017). [*When Cyclic Coordinate Descent Outperforms Randomized Coordinate Descent*.](https://papers.nips.cc/paper/7275-when-cyclic-coordinate-descent-outperforms-randomized-coordinate-descent.pdf) In Advances in Neural Information Processing Systems (pp. 7002-7010).

[23]: Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for generalized linear models via coordinate descent. Journal of statistical software, 33(1), 1.

[24]: Dünner, C., Parnell, T., Sarigiannis, D., Ioannou, N., & Pozidis, H. (2018). [*Snap Machine Learning*](https://arxiv.org/pdf/1803.06333.pdf). arXiv preprint arXiv:1803.06333.

[25]:  Rambler Digital Solutions. 2017. [*criteo-1tb-benchmark*](https://github.com/rambler-digital-solutions/criteo-1tb-benchmark). (2017). 

[26]: Andreas Sterbenz. 2017. [*Using Google Cloud Machine Learning to predict clicks at scale.*](https://cloud.google.com/blog/big-data/2017/02/using-google-cloud-machine-learning-to-predict-clicks-at-scale) Retrieved from Google.

[27]: Umbertogriffo (2018). [*Apache Spark - Best Practices and Tuning*](https://legacy.gitbook.com/book/umbertogriffo/apache-spark-best-practices-and-tuning/details). Retrieved from Gitbook.