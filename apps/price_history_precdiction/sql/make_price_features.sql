drop table if exists research.prices;
create table research.prices
as (
	with first_month as (
		select 
			company_id,
			case 
				when to_char(min(date) + '2 month'::interval, 'fmday') = 'saturday' then min(date) + '2 month 2 days'::interval
			  	when to_char(min(date) + '2 month'::interval, 'fmday') = 'sunday' then min(date) + '2 month 1 day'::interval
			  	else min(date) + '2 month'::interval
			end as first_two_month
		from stocks
		group by company_id 
	),
	
	stocks_sector_next as (
		select
			sg.*,
			c.gics_sector as sector,
			case 
				when to_char(date - '1 month'::interval, 'fmday') = 'saturday' then date - '1 month 1 day'::interval
			  	when to_char(date - '1 month'::interval, 'fmday') = 'sunday' then date - '1 month 2 days'::interval
			  	else date - '1 month'::interval
			end as prev_month
		from research.stock_growth sg
		left join first_month fm on fm.company_id = sg.company_id 
		left join companies c on c.company_id = sg.company_id
		where sg.date >= fm.first_two_month
	)
	
	select
		sn.company_id,
		sn.date as prediction_day,
		sn.ratio_prevmonth as stock_nextmonth_growth,
		stg.ratio_prevmonth as stock_prevmonth_growth,
		stg.ratio_prevsemiyear as stock_prevsemiyear_growth,
		stg.ratio_prevyear as stock_prevyear_growth,
		stg.ratio_prevfiveyear as stock_prevfiveyear_growth,
		seg.ratio_prevmonth as sector_prevmonth_growth,
		seg.ratio_prevsemiyear as sector_prevsemiyear_growth,
		seg.ratio_prevyear as sector_prevyear_growth,
		seg.ratio_prevfiveyear as sector_prevfiveyear_growth,
		stma.fifty_days_avg as stock_fifty_days_avg,
		stma.hundred_days_avg as stock_hundred_days_avg,
		stma.two_hundred_days_avg as stock_two_hundred_days_avg,
		sema.fifty_days_avg as sector_fifty_days_avg,
		sema.hundred_days_avg as sector_hundred_days_avg,
		sema.two_hundred_days_avg as sector_two_hundred_days_avg
	from stocks_sector_next sn
	left join research.stock_growth stg on stg.company_id  = sn.company_id and stg.date = sn.prev_month
	left join research.sector_growth seg on seg.sector = sn.sector and seg.date = sn.prev_month 
	left join research.stock_moving_average stma on stma.company_id  = sn.company_id and stma.date = sn.prev_month
	left join research.sector_moving_average sema on sema.sector  = sn.sector and sema.date = sn.prev_month
)