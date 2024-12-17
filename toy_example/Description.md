I want to make a toy example of a database for a Bookkeeping Store to manage the inventory, including the ability to handle excess and obsolete inventory.

**Tables and Relationships:**

1. **BOOKS**: This table stores information about each book available in the store.
2. **INVENTORY**: This table keeps track of the quantity of each book on hand, reorder levels, and warehouse locations.
3. **INVENTORY_ADJUSTMENTS**: This table records any adjustments made to the inventory, such as new shipments or obsolescence write-offs.

**Relationships:**

- Each book in the `BOOKS` table has one corresponding inventory record in the `INVENTORY` table.
- A book can have zero or more inventory adjustments recorded in the `INVENTORY_ADJUSTMENTS` table.

```mermaid
erDiagram
    BOOKS ||--|| INVENTORY : "has"
    BOOKS ||--o{ INVENTORY_ADJUSTMENTS : "has adjustments"

    BOOKS {
        int BookID PK
        string Title
        string Author
        string ISBN
        string Category
        string Publisher
        date PublicationDate
        float Price
    }

    INVENTORY {
        int BookID PK
        int QuantityOnHand
        int ReorderLevel
        string Status
        string WarehouseLocation
    }

    INVENTORY_ADJUSTMENTS {
        int AdjustmentID PK
        int BookID FK
        date AdjustmentDate
        string AdjustmentType
        int QuantityAdjusted
        string Reason
    }
