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
```


With this correction, the Mermaid ER diagram should parse correctly and visually represent the database structure you described.

**Explanation of the Tables:**

- **BOOKS**:
  - **BookID**: A unique identifier for each book (Primary Key).
  - **Title**, **Author**, **ISBN**, **Category**, **Publisher**, **PublicationDate**, **Price**: Additional details about the book.

- **INVENTORY**:
  - **BookID**: References the `BookID` from `BOOKS` (Primary Key and Foreign Key implicitly through the relationship).
  - **QuantityOnHand**: The current stock level.
  - **ReorderLevel**: The stock level at which more units should be ordered.
  - **Status**: Indicates if the book is active, discontinued, or obsolete.
  - **WarehouseLocation**: The physical location of the book in the warehouse.

- **INVENTORY_ADJUSTMENTS**:
  - **AdjustmentID**: A unique identifier for each inventory adjustment (Primary Key).
  - **BookID**: References the `BookID` from `BOOKS` (Foreign Key).
  - **AdjustmentDate**: The date of the inventory adjustment.
  - **AdjustmentType**: The type of adjustment (e.g., addition, subtraction, write-off).
  - **QuantityAdjusted**: The number of units adjusted.
  - **Reason**: A description or reason for the adjustment.

**Note:** In ER diagrams using Mermaid, when you define relationships between entities, the foreign key relationships are implicitly understood through those relationships. Therefore, you don't need to annotate `FK` on attributes already involved in relationships, especially when they are also `PK`. This is why removing the `FK` from `int BookID PK FK` resolves the parsing error.

