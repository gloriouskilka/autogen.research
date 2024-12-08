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
    LEFT JOIN INVENTORY_ADJUSTMENTS a
        ON i.BookID = a.BookID
        AND a.AdjustmentType = 'Sale'
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
- **LEFT JOIN** with `INVENTORY_ADJUSTMENTS` filtered for 'Sale' adjustments to find the last sale date.
- **GROUP BY** to aggregate data per book.
- **HAVING** clause to filter:
   1. Books explicitly marked as 'Obsolete'.
   2. Books with stock but no sales in the last 6 months.
*/
