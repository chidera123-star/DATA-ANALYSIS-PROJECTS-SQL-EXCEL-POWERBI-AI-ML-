-- ============================================================
--  BRAZILIAN E-COMMERCE DATASET — ANALYSIS QUERIES
--  Continues from: Cleaning_the_Brazilian_E-Commerce_Dataset.sql
--  Environment: MySQL
--  Tables used: olist_cleaned_orders, olist_order_payments_dataset,
--               olist_order_reviews_dataset
-- ============================================================


-- ============================================================
--  SECTION 1: SALES & REVENUE TRENDS
-- ============================================================

-- 1.1 Monthly Revenue & Order Volume
--      Tracks GMV and order count over time to spot seasonality
SELECT
    DATE_FORMAT(order_purchase_timestamp, '%Y-%m')  AS month,
    COUNT(DISTINCT order_id)                        AS total_orders,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue,
    ROUND(AVG(p.payment_value), 2)                  AS avg_order_value
FROM olist_cleaned_orders AS o
JOIN olist_order_payments_dataset AS p
    ON o.order_id = p.order_id
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY 1 ASC;


-- 1.2 Revenue by Day of Week
--      Identifies which days drive the most sales
SELECT
    DAYNAME(order_purchase_timestamp)               AS day_of_week,
    DAYOFWEEK(order_purchase_timestamp)             AS day_num,   -- for sorting
    COUNT(DISTINCT order_id)                        AS total_orders,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue
FROM olist_cleaned_orders AS o
JOIN olist_order_payments_dataset AS p
    ON o.order_id = p.order_id
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1, 2
ORDER BY 2 ASC;


-- 1.3 Payment Method Breakdown
--      Shows which payment types customers prefer and their avg values
SELECT
    p.payment_type,
    COUNT(*)                                        AS total_transactions,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue,
    ROUND(AVG(p.payment_value), 2)                  AS avg_payment_value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_transactions
FROM olist_order_payments_dataset AS p
JOIN olist_cleaned_orders AS o
    ON p.order_id = o.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY total_revenue DESC;


-- 1.4 Installment Usage
--      Reveals how many customers split payments and up to how many installments
SELECT
    p.payment_installments,
    COUNT(*)                                        AS order_count,
    ROUND(AVG(p.payment_value), 2)                  AS avg_order_value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_orders
FROM olist_order_payments_dataset AS p
JOIN olist_cleaned_orders AS o
    ON p.order_id = o.order_id
WHERE o.order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY 1 ASC;


-- 1.5 Year-over-Year Revenue Growth
SELECT
    YEAR(order_purchase_timestamp)                  AS year,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue,
    LAG(ROUND(SUM(p.payment_value), 2))
        OVER (ORDER BY YEAR(order_purchase_timestamp)) AS prev_year_revenue,
    ROUND(
        (SUM(p.payment_value) - LAG(SUM(p.payment_value))
            OVER (ORDER BY YEAR(order_purchase_timestamp)))
        / LAG(SUM(p.payment_value))
            OVER (ORDER BY YEAR(order_purchase_timestamp)) * 100
    , 2)                                            AS yoy_growth_pct
FROM olist_cleaned_orders AS o
JOIN olist_order_payments_dataset AS p
    ON o.order_id = p.order_id
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY 1 ASC;


-- ============================================================
--  SECTION 2: DELIVERY PERFORMANCE & LOGISTICS
-- ============================================================

-- 2.1 On-Time vs Late Delivery Rate
--      Core logistics KPI — what % of orders arrived on or before estimate
SELECT
    order_status,
    COUNT(*)                                        AS total_orders,
    SUM(CASE
            WHEN order_delivered_customer_date <= order_estimated_delivery_date
            THEN 1 ELSE 0
        END)                                        AS on_time_deliveries,
    SUM(CASE
            WHEN order_delivered_customer_date > order_estimated_delivery_date
            THEN 1 ELSE 0
        END)                                        AS late_deliveries,
    ROUND(
        SUM(CASE WHEN order_delivered_customer_date <= order_estimated_delivery_date
                 THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    )                                               AS on_time_rate_pct
FROM olist_cleaned_orders
WHERE order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
GROUP BY 1;


-- 2.2 Average Delivery Days by Customer State
--      Highlights which states have slowest/fastest delivery — useful for logistics planning
SELECT
    customer_state,
    COUNT(DISTINCT order_id)                        AS total_orders,
    ROUND(AVG(delivery_days), 1)                    AS avg_delivery_days,
    ROUND(MIN(delivery_days), 1)                    AS min_delivery_days,
    ROUND(MAX(delivery_days), 1)                    AS max_delivery_days
FROM olist_cleaned_orders
WHERE delivery_days IS NOT NULL
  AND delivery_days > 0
GROUP BY 1
ORDER BY avg_delivery_days DESC;


-- 2.3 Delivery Days Distribution (buckets)
--      Shows spread of delivery speed across all orders
SELECT
    CASE
        WHEN delivery_days <= 3  THEN '1–3 days (Express)'
        WHEN delivery_days <= 7  THEN '4–7 days (Fast)'
        WHEN delivery_days <= 14 THEN '8–14 days (Standard)'
        WHEN delivery_days <= 21 THEN '15–21 days (Slow)'
        ELSE '22+ days (Very Slow)'
    END                                             AS delivery_bucket,
    COUNT(*)                                        AS order_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_orders
FROM olist_cleaned_orders
WHERE delivery_days IS NOT NULL AND delivery_days > 0
GROUP BY 1
ORDER BY MIN(delivery_days) ASC;


-- 2.4 Late Deliveries & Their Average Review Score
--      Tests whether late delivery correlates with lower customer satisfaction
SELECT
    CASE
        WHEN order_delivered_customer_date <= order_estimated_delivery_date
        THEN 'On Time'
        ELSE 'Late'
    END                                             AS delivery_status,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    ROUND(AVG(r.review_score), 2)                   AS avg_review_score,
    ROUND(AVG(o.delivery_days), 1)                  AS avg_delivery_days
FROM olist_cleaned_orders AS o
JOIN olist_order_reviews_dataset AS r
    ON o.order_id = r.order_id
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY 1;


-- 2.5 Freight Value vs Delivery Days Correlation
--      Does higher freight cost actually mean faster delivery?
SELECT
    CASE
        WHEN freight_value < 10  THEN '< R$10'
        WHEN freight_value < 25  THEN 'R$10–25'
        WHEN freight_value < 50  THEN 'R$25–50'
        WHEN freight_value < 100 THEN 'R$50–100'
        ELSE 'R$100+'
    END                                             AS freight_bucket,
    COUNT(*)                                        AS order_count,
    ROUND(AVG(delivery_days), 1)                    AS avg_delivery_days,
    ROUND(AVG(freight_value), 2)                    AS avg_freight_value
FROM olist_cleaned_orders
WHERE delivery_days IS NOT NULL AND delivery_days > 0
  AND freight_value IS NOT NULL
GROUP BY 1
ORDER BY MIN(freight_value) ASC;


-- ============================================================
--  SECTION 3: CUSTOMER BEHAVIOUR & GEOGRAPHY
-- ============================================================

-- 3.1 Orders & Revenue by State
--      Geographic revenue distribution across Brazil
SELECT
    customer_state,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    COUNT(DISTINCT o.customer_id)                   AS unique_customers,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue,
    ROUND(AVG(p.payment_value), 2)                  AS avg_order_value
FROM olist_cleaned_orders AS o
JOIN olist_order_payments_dataset AS p
    ON o.order_id = p.order_id
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY total_revenue DESC;


-- 3.2 Top 15 Cities by Order Volume
SELECT
    customer_city,
    customer_state,
    COUNT(DISTINCT order_id)                        AS total_orders,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue
FROM olist_cleaned_orders AS o
JOIN olist_order_payments_dataset AS p
    ON o.order_id = p.order_id
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1, 2
ORDER BY total_orders DESC
LIMIT 15;


-- 3.3 Repeat vs One-Time Customers
--      Measures customer retention — what share of customers ordered more than once
SELECT
    purchase_frequency,
    COUNT(*)                                        AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_customers
FROM (
    SELECT
        customer_id,
        CASE
            WHEN COUNT(DISTINCT order_id) = 1 THEN 'One-Time'
            WHEN COUNT(DISTINCT order_id) = 2 THEN '2 Orders'
            WHEN COUNT(DISTINCT order_id) <= 5 THEN '3–5 Orders'
            ELSE '6+ Orders'
        END AS purchase_frequency
    FROM olist_cleaned_orders
    WHERE order_status NOT IN ('canceled', 'unavailable')
    GROUP BY customer_id
) AS freq_table
GROUP BY 1
ORDER BY MIN(purchase_frequency) ASC;


-- 3.4 Customer Review Score Distribution
SELECT
    r.review_score,
    COUNT(*)                                        AS review_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS pct_of_reviews
FROM olist_order_reviews_dataset AS r
JOIN olist_cleaned_orders AS o
    ON r.order_id = o.order_id
GROUP BY 1
ORDER BY 1 DESC;


-- 3.5 Average Review Score by State
--      Identifies which regions have the most/least satisfied customers
SELECT
    o.customer_state,
    ROUND(AVG(r.review_score), 2)                   AS avg_review_score,
    COUNT(DISTINCT o.order_id)                      AS total_orders
FROM olist_cleaned_orders AS o
JOIN olist_order_reviews_dataset AS r
    ON o.order_id = r.order_id
GROUP BY 1
HAVING total_orders > 50                           -- filter low-sample states
ORDER BY avg_review_score DESC;


-- ============================================================
--  SECTION 4: PRODUCT CATEGORY PERFORMANCE
-- ============================================================

-- 4.1 Revenue & Orders by Product Category
--      Top-performing categories by GMV
SELECT
    COALESCE(product_category_name_english, 'Unknown') AS category,
    COUNT(DISTINCT order_id)                        AS total_orders,
    ROUND(SUM(p.payment_value), 2)                  AS total_revenue,
    ROUND(AVG(p.payment_value), 2)                  AS avg_order_value,
    ROUND(AVG(freight_value), 2)                    AS avg_freight_cost
FROM olist_cleaned_orders AS o
JOIN olist_order_payments_dataset AS p
    ON o.order_id = p.order_id
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
ORDER BY total_revenue DESC
LIMIT 20;


-- 4.2 Category Delivery Performance
--      Which product categories tend to arrive fastest/slowest
SELECT
    COALESCE(product_category_name_english, 'Unknown') AS category,
    COUNT(DISTINCT order_id)                        AS total_orders,
    ROUND(AVG(delivery_days), 1)                    AS avg_delivery_days,
    ROUND(AVG(freight_value), 2)                    AS avg_freight_value,
    SUM(CASE
            WHEN order_delivered_customer_date > order_estimated_delivery_date
            THEN 1 ELSE 0
        END)                                        AS late_deliveries,
    ROUND(
        SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date
                 THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    )                                               AS late_rate_pct
FROM olist_cleaned_orders
WHERE delivery_days IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
GROUP BY 1
HAVING total_orders > 100
ORDER BY late_rate_pct DESC
LIMIT 20;


-- 4.3 Category Review Scores
--      Customer satisfaction by product type
SELECT
    COALESCE(o.product_category_name_english, 'Unknown') AS category,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    ROUND(AVG(r.review_score), 2)                   AS avg_review_score,
    SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END) AS low_score_count,
    ROUND(
        SUM(CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    )                                               AS low_score_pct
FROM olist_cleaned_orders AS o
JOIN olist_order_reviews_dataset AS r
    ON o.order_id = r.order_id
GROUP BY 1
HAVING total_orders > 100
ORDER BY avg_review_score DESC
LIMIT 20;


-- 4.4 Freight Cost as % of Order Value by Category
--      High freight-to-value ratio signals potential pricing or logistics issues
SELECT
    COALESCE(product_category_name_english, 'Unknown') AS category,
    COUNT(DISTINCT o.order_id)                      AS total_orders,
    ROUND(AVG(freight_value), 2)                    AS avg_freight,
    ROUND(AVG(p.payment_value), 2)                  AS avg_order_value,
    ROUND(AVG(freight_value) / NULLIF(AVG(p.payment_value), 0) * 100, 1) AS freight_pct_of_order
FROM olist_cleaned_orders AS o
JOIN olist_order_payments_dataset AS p
    ON o.order_id = p.order_id
WHERE order_status NOT IN ('canceled', 'unavailable')
GROUP BY 1
HAVING total_orders > 100
ORDER BY freight_pct_of_order DESC
LIMIT 20;


-- ============================================================
--  SECTION 5: EXECUTIVE SUMMARY VIEW
--  A single query combining top-level KPIs for a dashboard/report
-- ============================================================

SELECT
    COUNT(DISTINCT order_id)                        AS total_orders,
    COUNT(DISTINCT customer_id)                     AS total_customers,
    COUNT(DISTINCT product_id)                      AS unique_products,
    ROUND(AVG(delivery_days), 1)                    AS avg_delivery_days,
    SUM(CASE
            WHEN order_delivered_customer_date <= order_estimated_delivery_date
            THEN 1 ELSE 0
        END)                                        AS on_time_count,
    ROUND(
        SUM(CASE WHEN order_delivered_customer_date <= order_estimated_delivery_date
                 THEN 1 ELSE 0 END)
        * 100.0 / COUNT(*), 2
    )                                               AS on_time_rate_pct,
    MIN(order_purchase_timestamp)                   AS dataset_start,
    MAX(order_purchase_timestamp)                   AS dataset_end
FROM olist_cleaned_orders
WHERE order_status NOT IN ('canceled', 'unavailable')
  AND order_delivered_customer_date IS NOT NULL;
