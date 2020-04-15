drop table if exists research.stock_growth;
create table research.stock_growth as (
	with time_stamps as (
		select 
			*,
			case 
				when to_char(date - '1 month'::interval, 'fmday') = 'saturday' then date - '1 month 1 day'::interval
			  	when to_char(date - '1 month'::interval, 'fmday') = 'sunday' then date - '1 month 2 days'::interval
			  else date - '1 month'::interval
			end as prevmonth,
			case 
				when to_char(date - '6 month'::interval, 'fmday') = 'saturday' then date - '6 month 1 day'::interval
			  	when to_char(date - '6 month'::interval, 'fmday') = 'sunday' then date - '6 month 2 days'::interval
			  else date - '6 month'::interval
			end as prevsemiyear,
			case 
				when to_char(date - '1 year'::interval, 'fmday') = 'saturday' then date - '1 year 1 day'::interval
			  	when to_char(date - '1 year'::interval, 'fmday') = 'sunday' then date - '1 year 2 days'::interval
			  else date - '1 year'::interval
			end as prevyear,
			case 
				when to_char(date - '5 year'::interval, 'fmday') = 'saturday' then date - '5 year 1 day'::interval
			  	when to_char(date - '5 year'::interval, 'fmday') = 'sunday' then date - '5 year 2 days'::interval
			  else date - '5 year'::interval
			end as prevfiveyear
		from stocks
	)
	
	select
		ts.id,
		ts.company_id,
		ts.date,
		ts.close as cur_close,
		p1.date as prevmonth,
		p1.close as prevmonth_close,
		(ts.close - p1.close) / p1.close as ratio_prevmonth,
		p2.date as prevsemiyear,
		p2.close as prevsemiyear_close,
		(ts.close - p2.close) / p2.close as ratio_prevsemiyear,
		p3.date as prevyear,
		p3.close as prevyear_close,
		(ts.close - p3.close) / p3.close as ratio_prevyear,
		p4.date as prevfiveyear,
		p4.close as prevfiveyear_close,
		(ts.close - p4.close) / p4.close as ratio_prevfiveyear
	from time_stamps ts
	left join stocks p1 on p1.company_id = ts.company_id and p1.date = ts.prevmonth
	left join stocks p2 on p2.company_id = ts.company_id and p2.date = ts.prevsemiyear
	left join stocks p3 on p3.company_id = ts.company_id and p3.date = ts.prevyear
	left join stocks p4 on p4.company_id = ts.company_id and p4.date = ts.prevfiveyear
);


drop table if exists research.sector_growth;
create table research.sector_growth as (
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
	),
	
	time_stamps as (
		select 
			*,
			case 
				when to_char(date - '1 month'::interval, 'fmday') = 'saturday' then date - '1 month 1 day'::interval
			  	when to_char(date - '1 month'::interval, 'fmday') = 'sunday' then date - '1 month 2 days'::interval
			  else date - '1 month'::interval
			end as prevmonth,
			case 
				when to_char(date - '6 month'::interval, 'fmday') = 'saturday' then date - '6 month 1 day'::interval
			  	when to_char(date - '6 month'::interval, 'fmday') = 'sunday' then date - '6 month 2 days'::interval
			  else date - '6 month'::interval
			end as prevsemiyear,
			case 
				when to_char(date - '1 year'::interval, 'fmday') = 'saturday' then date - '1 year 1 day'::interval
			  	when to_char(date - '1 year'::interval, 'fmday') = 'sunday' then date - '1 year 2 days'::interval
			  else date - '1 year'::interval
			end as prevyear,
			case 
				when to_char(date - '5 year'::interval, 'fmday') = 'saturday' then date - '5 year 1 day'::interval
			  	when to_char(date - '5 year'::interval, 'fmday') = 'sunday' then date - '5 year 2 days'::interval
			  else date - '5 year'::interval
			end as prevfiveyear
		from sectors
	)

	select 
		ts.sector,
		ts.date,
		ts.sector_close as cur_close,
		p1.date as prevmonth,
		p1.sector_close as prevmonth_close,
		(ts.sector_close - p1.sector_close) / p1.sector_close as ratio_prevmonth,
		p2.date as prevsemiyear,
		p2.sector_close as prevsemiyear_close,
		(ts.sector_close - p2.sector_close) / p2.sector_close as ratio_prevsemiyear,
		p3.date as prevyear,
		p3.sector_close as prevyear_close,
		(ts.sector_close - p3.sector_close) / p3.sector_close as ratio_prevyear,
		p4.date as prevfiveyear,
		p4.sector_close as prevfiveyear_close,
		(ts.sector_close - p4.sector_close) / p4.sector_close as ratio_prevfiveyear
	from time_stamps ts
	left join sectors p1 on p1.sector = ts.sector and p1.date = ts.prevmonth
	left join sectors p2 on p2.sector = ts.sector and p2.date = ts.prevsemiyear
	left join sectors p3 on p3.sector = ts.sector and p3.date = ts.prevyear
	left join sectors p4 on p4.sector = ts.sector and p4.date = ts.prevfiveyear
);
