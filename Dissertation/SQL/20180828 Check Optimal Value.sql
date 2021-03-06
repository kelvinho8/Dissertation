
drop table #optimal
select 
rank() OVER(PARTITION BY model_name, train_start ORDER BY accuracy_score desc) as rnk
, *
into #optimal
FROM [DBHKUDissertation].[dbo].[GridSearchResult]
order by accuracy_score desc



select *
from #optimal
where rnk = 1
order by train_start, accuracy_score desc


select train_start, model_name, no_of_steps, window_size, max(accuracy_score) as max_accuracy_score
FROM [DBHKUDissertation].[dbo].[GridSearchResult]
group by train_start, model_name, no_of_steps, window_size 
order by train_start, model_name, no_of_steps, window_size




select *
into [DBHKUDissertation].[dbo].[GridSearchResult2008_2009]
FROM [DBHKUDissertation].[dbo].[GridSearchResult]
where train_start = '2008-01-01'



select *
from [DBHKUDissertation].[dbo].[GridSearchResult2008_2009]