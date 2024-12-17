-- Identify Excess Inventory
-- Criteria:
-- 1. QuantityOnHand is more than 1.5 times the ReorderLevel.
-- 2. Total sales are less than half of the current QuantityOnHand.

SELECT
    i.BookID,
    b.Title,
    i.QuantityOnHand,
    i.ReorderLevel,
    COALESCE(SUM(CASE WHEN a.AdjustmentType = 'Sale' THEN -a.QuantityAdjusted ELSE 0 END), 0) AS TotalSales
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
    AND COALESCE(SUM(-a.QuantityAdjusted), 0) < (i.QuantityOnHand / 2)
ORDER BY
    i.BookID;

 /*
 Explanation:
 - COALESCE ensures that books with no sales are treated as having zero sales.
 - Adjusted `QuantityAdjusted` sign to reflect actual sales quantity.
 */

/*
Explanation:
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
