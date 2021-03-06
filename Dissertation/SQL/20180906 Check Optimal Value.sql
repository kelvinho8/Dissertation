/****** Script for SelectTopNRows command from SSMS  ******/
SELECT TOP (1000) [dimensions]
      ,[interpolation_method]
      ,[interval]
      ,[model_name]
      ,[n_splits]
      ,[no_of_steps]
      ,[seed]
      ,[test_F1_score]
      ,[test_accuracy_score]
      ,[test_end]
      ,[test_precision_score]
      ,[test_recall_score]
      ,[test_roc_fn]
      ,[test_roc_fp]
      ,[test_roc_tn]
      ,[test_roc_tp]
      ,[test_start]
      ,[train_F1_score]
      ,[train_accuracy_score]
      ,[train_end]
      ,[train_precision_score]
      ,[train_recall_score]
      ,[train_roc_fn]
      ,[train_roc_fp]
      ,[train_roc_tn]
      ,[train_roc_tp]
      ,[train_start]
      ,[window_size]
      ,[x_test_end_date]
      ,[x_test_start_date]
      ,[x_train_end_date]
      ,[x_train_start_date]
      ,[y_test_false]
      ,[y_test_ratio]
      ,[y_test_true]
      ,[y_train_false]
      ,[y_train_ratio]
      ,[y_train_true]
  FROM [DBHKUDissertation].[dbo].[GridSearchResult]




  FROM [DBHKUDissertation].[dbo].[GridSearchResult]



  
drop table #optimal
select 
rank() OVER(PARTITION BY model_name, train_start ORDER BY train_accuracy_score desc) as train_rnk
, rank() OVER(PARTITION BY model_name, valid_start ORDER BY valid_accuracy_score desc) as valid_rnk
, rank() OVER(PARTITION BY model_name, test_start ORDER BY test_accuracy_score desc) as test_rnk

, *
into #optimal
FROM [DBHKUDissertation].[dbo].[GridSearchResult]
order by train_accuracy_score desc



select *
from #optimal
where train_rnk = 1
order by train_start, train_accuracy_score desc


select *
from #optimal
where valid_rnk = 1
order by valid_start, valid_accuracy_score desc



select *
from #optimal
where test_rnk = 1
order by test_start, test_accuracy_score desc



--select train_start, model_name, no_of_steps, window_size, max(train_accuracy_score) as max_train_accuracy_score
--FROM [DBHKUDissertation].[dbo].[GridSearchResult]
--group by train_start, model_name, no_of_steps, window_size 