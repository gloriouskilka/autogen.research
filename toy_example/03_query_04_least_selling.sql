-- Identify Least-Selling Books with Sales Data
SELECT
    b.BookID,
    b.Title,
    COALESCE(SUM(-a.QuantityAdjusted), 0) AS TotalUnitsSold,
    COALESCE(SUM(-a.QuantityAdjusted) * b.Price, 0) AS TotalRevenue
FROM
    BOOKS b
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID AND a.AdjustmentType = 'Sale'
GROUP BY
    b.BookID,
    b.Title,
    b.Price
HAVING
    TotalUnitsSold < 10 -- Threshold can be adjusted based on overall sales volume
ORDER BY
    TotalUnitsSold ASC
LIMIT 10;

 /*
 Explanation:
 - Identifies books with fewer than a specified number of units sold.
 - Helps in pinpointing items that may require promotional efforts or reevaluation.

 Actionable Steps:
Run Promotions: Implement discounts or bundle deals to boost sales.
Evaluate Continuation: Consider discontinuing persistently low-selling titles to free up inventory space.
Assess Pricing: Review pricing strategies for low-performing books
 */
