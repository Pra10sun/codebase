drop table if exists research.stock_moving_average;
create table research.stock_moving_average as
select
	company_id,
	date,
	open,
	high,
	low,
	close,
	adjusted_close,
	volume,
	dividend_amount,
	avg(close) over(
		partition by company_id order by date 
		rows between 50 preceding and current row
	) as fifty_days_avg,
	avg(close) over(
		partition by company_id order by date 
		rows between 100 preceding and current row
	) as hundred_days_avg,
	avg(close) over(
		partition by company_id order by date 
		rows between 200 preceding and current row
	) as two_hundred_days_avg
from stocks;


drop table if exists research.sector_moving_average;
create table research.sector_moving_average as (
	with sectors as (
		select 
			gics_sector as sector,
			s.date as date,
			sum(open) as sector_open,
			sum(high) as sector_high,
			sum(low) as sector_low,
			sum(close) as sector_close,
			sum(adjusted_close) as sector_adjusted_close,
			sum(volume) as sector_volume,
			sum(dividend_amount) as sector_dividend
		from stocks s
		join companies c
		on c.company_id = s.company_id
		group by c.gics_sector, s.date
	)
	
	select
		sector,
		date,
		sector_open,
		sector_high,
		sector_low,
		sector_close,
		sector_adjusted_close,
		sector_volume,
		sector_dividend,
		avg(sector_close) over(
			partition by sector order by date 
			rows between 50 preceding and current row
		) as fifty_days_avg,
		avg(sector_close) over(
			partition by sector order by date 
			rows between 100 preceding and current row
		) as hundred_days_avg,
		avg(sector_close) over(
			partition by sector order by date 
			rows between 200 preceding and current row
		) as two_hundred_days_avg
	from sectors
);
