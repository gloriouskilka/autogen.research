-- Identify Seasonal Sales Trends
-- Example: Analyze sales by month to identify peak seasons.

SELECT
    MONTH(a.AdjustmentDate) AS SaleMonth,
    COUNT(a.AdjustmentID) AS TotalSalesTransactions,
    SUM(-a.QuantityAdjusted) AS TotalUnitsSold,
    SUM(-a.QuantityAdjusted * b.Price) AS TotalSalesRevenue
FROM
    INVENTORY_ADJUSTMENTS a
    JOIN BOOKS b ON a.BookID = b.BookID
WHERE
    a.AdjustmentType = 'Sale'
    AND a.AdjustmentDate >= DATE_SUB(CURDATE(), INTERVAL 2 YEAR)
GROUP BY
    YEAR(a.AdjustmentDate),
    MONTH(a.AdjustmentDate)
ORDER BY
    YEAR(a.AdjustmentDate),
    MONTH(a.AdjustmentDate);

 /*
 Explanation:
 - Aggregates sales data by month over the past two years.
 - Identifies months with higher or lower sales volumes.
 - Facilitates planning for seasonal demands and inventory adjustments.

Actionable Steps:
Prepare for High-Demand Seasons: Increase stock levels and staffing during identified peak periods.
Promote During Low-Demand Periods: Implement marketing strategies to boost sales during traditionally slow months.
Plan Events and Promotions: Align special events, sales, or promotions with seasonal trends to maximize impact.
 */
