## Overview

opencellid dataset for **Senegal** with 90-day observation window

a subset of of OpenCellID with cellular network infrastructure that provides geospatial and technical information about mobile cell towers across **Senegal** with 90-day observation window

## Data Source:

[Cell Towers Worldwide: Location Data by Continent on Kaggle](https://www.kaggle.com/datasets/zakariaeyoussefi/cell-towers-worldwide-location-data-by-continent)

This extensive dataset provides geographic coordinates and network information for cell tower locations across the globe, organized by continent. It includes the following columns:

* Radio: The generation of broadband cellular network technology (e.g., LTE, GSM).  
* MCC: Mobile Country Code, a unique identifier for each country in the mobile network.  
* MNC: Mobile Network Code, identifying the mobile network within a country.  
* LAC: Location Area Code, Tracking Area Code, or Network Identifier.  
* CID: Unique identifier for each Base Transceiver Station (BTS) or sector.  
* Longitude: Geographic coordinate specifying the east-west position.  
* Latitude: Geographic coordinate specifying the north-south position.  
* Range: Approximate area within which the cell coverage extends (in meters).  
* Samples: Number of measures processed to derive the data point.  
* Changeable: Indicates if the cell location was determined through sample processing (1) or directly obtained from the telecom firm (0).  
* Created: Timestamp indicating when the cell was first added to the database (UNIX format).  
* Updated: Timestamp indicating when the cell was last seen or updated in the database (UNIX format).  
* AverageSignal: Represents the averaged signal strength of the cell location.  
* Country: The country of the cell tower.  
* Network: The company that owns the cell tower.  
* Continent: The continent of the cell tower.

### Filtering

* Filter Senegal data from Africa (MCC=608).  
  * Mobile Country Code (MCC) for Senegal is 608  
4. Filter Express from Orange and Tiggo (net=3)  
   1. sMobile Network Code (MNC) for Expresso is 03  
5. 90-day observation window  
   1.  Consider 90 days to match the churn definition  
   2. Churn indicates whether a customer becomes inactive and makes no transactions for 90 consecutive days.

Based on [OpenCelliD](https://opencellid.org/)

## Sample records
```
radio	MCC	MNC	TAC	CID	unit	LON	LAT	RANGE	SAM	changeable	created	updated	averageSignal	Country	Network	Continent	created_date	updated_date

0	GSM	608	3	801	60053	0	-14.943466	12.890396	1000.0	5	1	1459813076	1460884030	0.0	Senegal	Expresso	Africa	2016-04-04 23:37:56	2016-04-17 09:07:10

1	GSM	608	3	401	43013	0	-16.500776	14.361879	2792.0	2	1	1351194533	1456970224	0.0	Senegal	Expresso	Africa	2012-10-25 19:48:53	2016-03-03 01:57:04

2	GSM	608	3	401	43011	0	-16.496656	14.361879	3218.0	2	1	1351194533	1456970224	0.0	Senegal	Expresso	Africa	2012-10-25 19:48:53	2016-03-03 01:57:04
```

# OpenCelliD

## Overview

OpenCellID is a global open database of cellular network infrastructure that provides geospatial and technical information about mobile cell towers across the world. 

The dataset consists of infrastructure-level variables describing network technology, geographic position, coverage characteristics, and observation timestamps.

It is updated on a daily basis and includes cell towers observed within the last 18 months.

## Data source

OpenCelliD   
Largest Open Database of Cell Towers & Geolocation by Unwired Labs  
Database format - OpenCellID wiki  
[https://wiki.opencellid.org/wiki/Database_format](https://wiki.opencellid.org/wiki/Database_format)

## Download 

The dataset can be downloaded from   
[https://opencellid.org/downloads](https://opencellid.org/downloads)

## Variable definitions

| Parameter | Data type | Description |
| :---- | :---- | :---- |
| radio | string | Network type. One of the strings GSM, UMTS, LTE or CDMA. |
| mcc | integer | Mobile Country Code, for example 260 for Poland. |
| net | integer | For GSM, UMTS and LTE networks, this is the Mobile Network Code (MNC). For CDMA networks, this is the System IDentification number (SID). |
| area | integer | Location Area Code (LAC) for GSM and UMTS networks. Tracking Area Code (TAC) for LTE networks. Network IDenfitication number (NID) for CDMA networks. |
| cell | integer | Cell ID (CID) for GSM and LTE networks. UTRAN Cell ID / LCID for UMTS networks, which is the concatenation of 2 or 4 bytes of Radio Network Controller (RNC) code and 4 bytes of Cell ID. Base station IDentifier number (BID) for CDMA networks. |
| unit | integer | Primary Scrambling Code (PSC) for UMTS networks. Physical Cell ID (PCI) for LTE networks. An empty value for GSM and CDMA networks. |
| lon | double | Longitude in degrees between -180.0 and 180.0. changeable=1: average of longitude values; changeable=0: exact GPS position. |
| lat | double | Latitude in degrees between -90.0 and 90.0. changeable=1: average of latitude values; changeable=0: exact GPS position. |
| range | integer | Estimate of cell range, in meters. |
| samples | integer | Total number of measurements assigned to the cell tower. |
| changeable | integer | Defines if coordinates are exact (0) or approximate/calculated (1). |
| created | integer | The first time the cell tower was added (Unix timestamp). For example, 1409522613\. |
| updated | integer | The last time the cell tower was updated (Unix timestamp). |
| averageSignal | integer | Average signal strength from all assigned measurements (dBm or TS 27.007 8.5). |

## Sample records
```
radio	mcc	net	area	cell	unit	lon	lat	range_val	samples	      changeable	created	updated	 averageSignal

LTE	608	1	7200	9985	0	-17.4709	14.7086	1000	13	1	1469357889	1747847643	0  
LTE	608	1	7200	26882	0	-17.4654	14.7142	1000	15	1	1469357889	1732646710	0  
LTE	608	1	7200	59905	0	-17.4669	14.7178	1763	50	1	1469357889	1745780044	0
```