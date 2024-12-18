-- Analyze Stock Aging
-- Identify how long each book has been in inventory based on the oldest inventory adjustment.

SELECT
    b.BookID,
    b.Title,
    i.QuantityOnHand,
    MIN(a.AdjustmentDate) AS FirstInventoryDate,
    DATEDIFF(CURDATE(), MIN(a.AdjustmentDate)) AS DaysInInventory
FROM
    BOOKS b
    JOIN INVENTORY i ON b.BookID = i.BookID
    JOIN INVENTORY_ADJUSTMENTS a ON b.BookID = a.BookID
GROUP BY
    b.BookID,
    b.Title,
    i.QuantityOnHand
ORDER BY
    DaysInInventory DESC;

 /*
 Explanation:
 - FirstInventoryDate: The date when the book was first added to inventory.
 - DaysInInventory: Number of days the book has been in inventory.
 - Older inventory items are listed first, highlighting potential stock that may need attention.

Actionable Steps:
Promote Aging Stock: Implement marketing strategies such as discounts, bundles, or special promotions to move older inventory.
Review Procurement Practices: Adjust ordering patterns to prevent accumulation of long-held stock.
Plan for Obsolescence: Proactively manage aging inventory to avoid future obsolescence write-offs.
 */
