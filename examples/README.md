# Examples

This directory contains examples and demonstrations for Flash Dynamic Mask Attention.

## Files

### `query_agnostic_demo.py`

A standalone demonstration that illustrates the query-agnostic nature of the dynamic masking mechanism. This script shows:

- How ZOH states are computed from Value vectors only
- How the same importance scores are broadcast to all queries
- How TopK selection produces the same keys for all queries
- The implications of this design for different types of tasks

**Run the demo:**
```bash
python examples/query_agnostic_demo.py
```

This demo helps understand the trade-offs between computational efficiency and query-specific precision discussed in Issue #117.

### `modeling/`

Contains example model implementations showing how to integrate Flash Dynamic Mask Attention into transformer architectures.

## Related Documentation

- [Design Choices](../docs/design_choices.md) - Detailed analysis of the query-agnostic design
- [Integration Guide](../docs/integration.md) - Technical implementation details
- [API Reference](../docs/api_reference.md) - Function documentation