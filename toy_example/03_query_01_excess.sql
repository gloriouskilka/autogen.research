-- Identify Excess Inventory
-- Criteria:
-- 1. QuantityOnHand is more than 1.5 times the ReorderLevel.
-- 2. Total sales are less than half of the current QuantityOnHand.

SELECT
    i.BookID,
    b.Title,
    i.QuantityOnHand,
    i.ReorderLevel,
    COALESCE(SUM(ABS(a.QuantityAdjusted)), 0) AS TotalUnitsSold
FROM
    INVENTORY i
    JOIN BOOKS b ON i.BookID = b.BookID
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON i.BookID = a.BookID AND a.AdjustmentType = 'Sale'
GROUP BY
    i.BookID,
    b.Title,
    i.QuantityOnHand,
    i.ReorderLevel
HAVING
    i.QuantityOnHand > (i.ReorderLevel * 1.5)
    AND COALESCE(SUM(ABS(a.QuantityAdjusted)), 0) < (i.QuantityOnHand / 2)
ORDER BY
    i.BookID;

/*
Explanation:
- Uses ABS to ensure QuantityAdjusted is treated as positive for sales.
- Renamed TotalSales to TotalUnitsSold for clarity.
*/

/*
Objective: Identify books with excess inventory by ensuring:
1. Current stock exceeds 1.5 times the reorder level.
2. Total units sold are less than half of the current stock, indicating slow movement.
*/


/*
Objective: Find books where the current stock is significantly higher than the reorder level, and sales are not sufficient to justify the high stock levels.
Logic:
Excess Stock Condition: QuantityOnHand > (ReorderLevel * 1.5)
Indicates that the stock on hand is 50% more than the reorder level.
Low Sales Condition: TotalSales < (QuantityOnHand / 2)
Total sales are less than half of the current stock, suggesting slow-moving inventory.
Columns Selected:
BookID, Title, QuantityOnHand, ReorderLevel, and TotalSales
Aggregations and Calculations:
Calculated TotalSales by summing the quantities adjusted for sales (AdjustmentType = 'Sale').
Results:
Based on the sample data, this query should identify BookID 3 ('Clean Code') as having excess inventory.
*/
