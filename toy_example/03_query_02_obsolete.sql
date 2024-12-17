-- Identify Obsolete Inventory
-- Criteria:
-- 1. Status is 'Obsolete'.
-- 2. OR QuantityOnHand > 0 AND no sales in the last 6 months.

SELECT
    i.BookID,
    b.Title,
    i.QuantityOnHand,
    i.Status,
    MAX(a.AdjustmentDate) AS LastSaleDate
FROM
    INVENTORY i
    JOIN BOOKS b ON i.BookID = b.BookID
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON i.BookID = a.BookID AND a.AdjustmentType = 'Sale'
GROUP BY
    i.BookID,
    b.Title,
    i.QuantityOnHand,
    i.Status
HAVING
    i.Status = 'Obsolete'
    OR (
        i.QuantityOnHand > 0
        AND (
            MAX(a.AdjustmentDate) IS NULL
            OR MAX(a.AdjustmentDate) < DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
        )
    )
ORDER BY
    i.BookID;

 /*
 Explanation:
 - Filters books marked as 'Obsolete'.
 - Additionally identifies books with stock but no sales in the last 6 months.
 */

/*
Explanation:
Objective: Find books that are obsolete or have poor sales indicating potential obsolescence.
Logic:
Obsolete Status: i.Status = 'Obsolete'
Immediately flags books marked as obsolete.
No Recent Sales Condition:
If the book has stock on hand but no sales in the last 6 months, it might be obsolete.
Columns Selected:
BookID, Title, QuantityOnHand, Status, and LastSaleDate
Aggregations and Calculations:
Calculated LastSaleDate as the most recent sale date (`AdjustmentType = 'Sale'`) for each book.
*/
