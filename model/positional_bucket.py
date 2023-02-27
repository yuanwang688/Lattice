import torch    

def compute_positional_bucket(query_length, key_length, type_ids, row_ids, col_ids, num_buckets):
    """ Compute positional bucket for the queries and keys in the Encoder. 

    Assume this is self-attention. Therefore, query_length=-key_length.

    Args:
      query_length: sequence length for queries.
      key_length: sequence length for keys. This should be the same as query_length.
      type_ids: type id of queries / keys
      row_ids: row id of the queries / keys
      col_ids: column ids of the queries / keys.
      num_buckets: this is the maximum number of buckets for the relative attention bias.

    Returns:
      Positional buckets: (batch_size, query_length, key_length) tensor of bucket indices.
        query and key are both in metadata or both in the same cell: -1
        query is in metadata and key is in a cell: num_buckets - 5
        query is in a cell and key is in metadata: num_buckets - 4
        query and key are in the same row: num_buckets - 3
        query and key are in the same col: num_buckets - 2
        query and key are not in the same row or col: num_buckets - 1
    """

    # shape (batch_size, query_length)
    is_meta = torch.logical_and(type_ids < 2.5, type_ids > 0.5)
    # shape (batch_size, query_length)
    is_cell = type_ids == 3 

    # meta_mask == 1 if both query and key are in metadata
    # shape (batch_size, query_length, key_length)
    meta_mask = torch.bmm(torch.unsqueeze(is_meta.float(), 2), torch.unsqueeze(is_meta.float(), 1)) > 0.5

    # cell_mask == 1 if both query and key are in the table (i.e. in some cell)
    # shape (batch_size, query_length, key_length)
    cell_mask = torch.bmm(torch.unsqueeze(is_cell.float(), 2), torch.unsqueeze(is_cell.float(), 1)) > 0.5

    # meta_to_cell_mask == 1 if query is a metadata and key is in a cell
    meta_to_cell_mask = torch.bmm(torch.unsqueeze(is_meta.float(), 2), torch.unsqueeze(is_cell.float(), 1)) > 0.5
    # cell_to_metadata == 1 if query is in a cell and key is in a metadata
    cell_to_meta_mask = torch.bmm(torch.unsqueeze(is_cell.float(), 2), torch.unsqueeze(is_meta.float(), 1)) > 0.5

    row_diff = torch.abs(row_ids.unsqueeze(-1) - row_ids.unsqueeze(1))  # shape (batch_size, query_length, key_length)
    col_diff = torch.abs(col_ids.unsqueeze(-1) - col_ids.unsqueeze(1))  # shape (batch_size, query_length, key_length)

    # same_cell_mask == 1 if both query and key are in the same cell
    same_cell_mask = torch.logical_and(row_diff + col_diff < 0.5, cell_mask)

    # same_row_mask == 1 if both query and key are in the same row
    same_row_mask = torch.logical_and(row_diff < 0.5, cell_mask)

    # same_col_mask == 1 if both query and key are in the same col
    same_col_mask = torch.logical_and(col_diff < 0.5, cell_mask)

    positional_buckets = (
    	-1 * torch.logical_or(meta_mask, same_cell_mask)
    	+ (num_buckets-5) * meta_to_cell_mask
    	+ (num_buckets-4) * cell_to_meta_mask
    	+ (num_buckets-3) * torch.logical_and(same_row_mask, torch.logical_not(same_cell_mask))
    	+ (num_buckets-2) * torch.logical_and(same_col_mask, torch.logical_not(same_cell_mask))
    	+ (num_buckets-1) * torch.logical_and(cell_mask, 
    		torch.logical_not(torch.logical_or(same_col_mask, same_row_mask)))
    	)

    return positional_buckets.long()


def main():

	# tokens: (type=metadata(m)/cell(c), row_id, col_id)
	# (m,0,0),(m,0,0),(m,1,0),(c,2,0),(c,2,0),(c,2,1),(c,3,0),(c,3,1)
	type_ids = torch.Tensor([[1, 1, 2, 3, 3, 3, 3, 3]])
	row_ids = torch.Tensor([[0, 0, 1, 2, 2, 2, 3, 3]])
	col_ids = torch.Tensor([[0, 0, 0, 0, 0, 1, 0, 1]])
	num_buckets = 100
	query_length = 8
	key_length = 8

	positional_buckets = compute_positional_bucket(
		query_length, key_length, type_ids, row_ids, col_ids, num_buckets)

	print(positional_buckets)

	expected_positional_buckets = torch.Tensor([[
		[-1, -1, -1, 95, 95, 95, 95, 95],
		[-1, -1, -1, 95, 95, 95, 95, 95],
		[-1, -1, -1, 95, 95, 95, 95, 95],
		[96, 96, 96, -1, -1, 97, 98, 99],
		[96, 96, 96, -1, -1, 97, 98, 99],
		[96, 96, 96, 97, 97, -1, 99, 98],
		[96, 96, 96, 98, 98, 99, -1, 97],
		[96, 96, 96, 99, 99, 98, 97, -1]]]).long()

	print(torch.equal(positional_buckets, expected_positional_buckets))


if __name__ == "__main__":
    main()
