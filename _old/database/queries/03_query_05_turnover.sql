-- Calculate Inventory Turnover Ratio
-- Formula: Inventory Turnover = Cost of Goods Sold (COGS) / Average Inventory

-- For simplicity, we'll use Total Sales as a proxy for COGS.
-- Note: In a real-world scenario, COGS may differ from sales revenue.

SELECT
    b.BookID,
    b.Title,
    COALESCE(SUM(-a.QuantityAdjusted), 0) AS TotalUnitsSold,
    COALESCE(SUM(-a.QuantityAdjusted * b.Price), 0) AS TotalSalesRevenue,
    i.QuantityOnHand,
    COALESCE(SUM(-a.QuantityAdjusted), 0) / NULLIF(i.QuantityOnHand, 0) AS InventoryTurnoverRatio
FROM
    BOOKS b
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID AND a.AdjustmentType = 'Sale'
    JOIN INVENTORY i ON b.BookID = i.BookID
GROUP BY
    b.BookID,
    b.Title,
    i.QuantityOnHand
ORDER BY
    InventoryTurnoverRatio DESC;

 /*
 Explanation:
 - TotalUnitsSold: Total number of units sold per book.
 - TotalSalesRevenue: Total revenue generated from sales per book.
 - InventoryTurnoverRatio: Indicates how many times the inventory is sold and replaced over the period.
 - NULLIF is used to prevent division by zero.
 - Higher ratios suggest better inventory management.

 Actionable Steps:
Improve Turnover for Low Ratios: For books with low turnover ratios, consider promotional activities, bundling, or discounting to increase sales.
Reassess Stock Levels: Adjust future purchase quantities based on turnover rates to optimize inventory levels.
Identify Best Performers: Focus on high-turnover items for potential expansion of stock or additional related products.
 */
