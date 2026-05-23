-- Step 1  Open MySQL Workbench

/*
Start MySQL Workbench
Connect to your MySQL server
Open a New SQL Tab
*/

-- Step 2  Create telecom_churn Database

CREATE DATABASE telecom_churn;

-- Step 3  Select Database

USE telecom_churn;

-- Step 4  Create opencellid Table
-- opencellid table stores cell tower location data.

CREATE TABLE opencellid (
    radio VARCHAR(10),
    mcc INT,
    net INT,
    area INT,
    cell BIGINT,
    unit INT,
    lon DOUBLE,
    lat DOUBLE,
    range_val INT,
    samples INT,
    changeable INT,
    created BIGINT,
    updated BIGINT,
    averageSignal INT
);

-- Step 5 — Create expresso_sample Table
-- expresso_sample table stores telecom customer data for churn analysis.

CREATE TABLE expresso_sample (
    user_id VARCHAR(100) PRIMARY KEY,
    region VARCHAR(50),
    tenure VARCHAR(30),
    montant DECIMAL(10,2),
    frequence_rech DECIMAL(8,2),
    revenue DECIMAL(10,2),
    arpu_segment DECIMAL(10,2),
    frequence DECIMAL(8,2),
    data_volume DECIMAL(12,2),
    on_net DECIMAL(12,2),
    orange DECIMAL(12,2),
    tigo DECIMAL(12,2),
    zone1 DECIMAL(12,2),
    zone2 DECIMAL(12,2),
    mrg VARCHAR(5),
    regularity INT,
    top_pack VARCHAR(150),
    freq_top_pack DECIMAL(8,2),
    churn TINYINT
);

-- Step 6 Set import configuration

SET GLOBAL local_infile = 1;

SHOW VARIABLES LIKE 'local_infile';
/*
# Variable_name	Value
local_infile	ON
*/

SHOW VARIABLES LIKE 'secure_file_priv';
/*
# Variable_name	Value
secure_file_priv	/path/to/mysql-files/
*/

-- Move csv files to: /path/to/mysql-files/


-- Step 7 Import opencellid Dataset

LOAD DATA INFILE '/path/to/mysql-files/opencellid_senegal.csv'
INTO TABLE opencellid
FIELDS TERMINATED BY ','
LINES TERMINATED BY '\n';

-- Step 8 — Import expresso Dataset
-- expresso Dataset contains missing values, so use this pattern: NULLIF(@column,'')

LOAD DATA INFILE '/path/to/mysql-files/expresso_sample_100k.csv'
INTO TABLE expresso_sample
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS
(
@user_id,
@region,
@tenure,
@montant,
@frequence_rech,
@revenue,
@arpu_segment,
@frequence,
@data_volume,
@on_net,
@orange,
@tigo,
@zone1,
@zone2,
@mrg,
@regularity,
@top_pack,
@freq_top_pack,
@churn
)
SET
user_id = @user_id,
region = NULLIF(@region,''),
tenure = NULLIF(@tenure,''),
montant = NULLIF(@montant,''),
frequence_rech = NULLIF(@frequence_rech,''),
revenue = NULLIF(@revenue,''),
arpu_segment = NULLIF(@arpu_segment,''),
frequence = NULLIF(@frequence,''),
data_volume = NULLIF(@data_volume,''),
on_net = NULLIF(@on_net,''),
orange = NULLIF(@orange,''),
tigo = NULLIF(@tigo,''),
zone1 = NULLIF(@zone1,''),
zone2 = NULLIF(@zone2,''),
mrg = NULLIF(@mrg,''),
regularity = NULLIF(@regularity,''),
top_pack = NULLIF(@top_pack,''),
freq_top_pack = NULLIF(@freq_top_pack,''),
churn = NULLIF(@churn,'');

-- Step 9 — Verify Data

SELECT COUNT(*) FROM opencellid; -- '2709'

SELECT COUNT(*) FROM expresso_sample; -- '100000'

-- Step 10 — Preview Data

SELECT * FROM opencellid LIMIT 1;

/*
radio	mcc	net	area	cell	unit	lon	lat	range_val	samples	changeable	created	updated	averageSignal
LTE	608	1	7200	9985	0	-17.4709	14.7086	1000	13	1	1469357889	1747847643	0
*/

SELECT * FROM expresso_sample LIMIT 1;

/*
user_id	region	tenure	montant	frequence_rech	revenue	arpu_segment	frequence	data_volume	on_net	orange	tigo	zone1	zone2	mrg	regularity	top_pack	freq_top_pack	churn
000028d9e13a595abe061f9b58f3d76ab907850f	DAKAR	K > 24 month	1000.00	1.00	985.00	328.00	1.00		39.00	24.00				NO	11	Mixt 250F=Unlimited_call24H	2.00	0
*/
