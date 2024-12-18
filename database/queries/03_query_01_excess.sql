-- Identify Excess Inventory
-- Criteria:
-- 1. QuantityOnHand is more than 1.5 times the ReorderLevel.
-- 2. Total sales are less than half of the current QuantityOnHand.

SELECT
    i.BookID,
    b.Title,
    i.QuantityOnHand,
    i.ReorderLevel,
    COALESCE(SUM(CASE WHEN a.AdjustmentType = 'Sale' THEN a.QuantityAdjusted ELSE 0 END), 0) AS TotalUnitsSold
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
    AND COALESCE(SUM(CASE WHEN a.AdjustmentType = 'Sale' THEN a.QuantityAdjusted ELSE 0 END), 0) < (i.QuantityOnHand / 2)
ORDER BY
    i.BookID;

/*
Explanation:
- **SUM with CASE Statement**: Explicitly sums only 'Sale' adjustments to calculate `TotalUnitsSold`.
- **COALESCE**: Ensures that books with no sales are treated as having zero sales.
- **Overall Logic**: Identifies books with excess stock and low sales velocity.
*/
