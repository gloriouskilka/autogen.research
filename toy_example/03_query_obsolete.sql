-- Identify Obsolete Inventory
-- We can identify obsolete books by checking their Status and analyzing recent sales activity.
-- Identify obsolete books
SELECT
    i.BookID,
    b.Title,
    i.QuantityOnHand,
    i.Status,
    MAX(CASE WHEN a.AdjustmentType = 'Sale' THEN a.AdjustmentDate ELSE NULL END) AS LastSaleDate
FROM
    INVENTORY i
    JOIN BOOKS b ON i.BookID = b.BookID
    LEFT JOIN INVENTORY_ADJUSTMENTS a ON i.BookID = a.BookID
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
            MAX(CASE WHEN a.AdjustmentType = 'Sale' THEN a.AdjustmentDate ELSE NULL END) IS NULL
            OR MAX(CASE WHEN a.AdjustmentType = 'Sale' THEN a.AdjustmentDate ELSE NULL END) < DATE_SUB(CURDATE(), INTERVAL 6 MONTH)
        )
    )
ORDER BY
    i.BookID;

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