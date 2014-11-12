package edu.pengli.nlp.platform.types;

import java.io.Serializable;
import java.util.Arrays;

/*
 * 	 A vector that allocates memory only for non-zero values.
 */
public class SparseVector implements Serializable {

	protected int[] indices; // if this is null, then the vector is dense
	protected double[] values; // if this is null, then the vector is binary

	public SparseVector(int[] indices, double[] values, int capacity, int size,
			boolean copy, boolean checkIndicesSorted, boolean removeDuplicates) {
		// "size" was pretty much ignored??? Why?
		int length;
		length = size;
		if (capacity < length)
			capacity = length;
		assert (size <= length);
		if (!(values == null || indices == null || indices.length == values.length))
			throw new IllegalArgumentException(
					"Attempt to create sparse non-binary SparseVector with mismatching values & indices\n"
							+ "  indices.length = "
							+ indices.length
							+ "   values.length = " + values.length);
		if (copy || capacity > length) {
			if (indices == null)
				this.indices = null;
			else {
				this.indices = new int[capacity];
				System.arraycopy(indices, 0, this.indices, 0, length);
			}
			if (values == null)
				this.values = null;
			else {
				this.values = new double[capacity];
				System.arraycopy(values, 0, this.values, 0, length);
			}
		} else {
			this.indices = indices;
			this.values = values;
		}
		if (checkIndicesSorted)
			sortIndices(); // This also removes duplicates
		else if (removeDuplicates)
			removeDuplicates(0);
	}

	public SparseVector(int[] indices, double[] values, boolean copy,
			boolean checkIndicesSorted, boolean removeDuplicates) {
		this(indices, values, (indices != null) ? indices.length
				: values.length, (indices != null) ? indices.length
				: values.length, copy, checkIndicesSorted, removeDuplicates);
	}

	public SparseVector(int[] indices, double[] values) {
		this(indices, values, true, true, true);
	}

	public SparseVector(int[] indices, double[] values, boolean copy) {
		this(indices, values, copy, true, true);
	}

	public SparseVector(int[] indices, double[] values, boolean copy,
			boolean checkIndicesSorted) {
		this(indices, values, copy, checkIndicesSorted, true);
	}

	// Create a vector that is possibly binary or non-binary
	public SparseVector(int[] indices, boolean copy,
			boolean checkIndicesSorted, boolean removeDuplicates, boolean binary) {
		this(indices, binary ? null : newArrayOfValue(indices.length, 1.0),
				indices.length, indices.length, copy, checkIndicesSorted,
				removeDuplicates);
	}

	// Create a binary vector
	public SparseVector(int[] indices, int capacity, int size, boolean copy,
			boolean checkIndicesSorted, boolean removeDuplicates) {
		this(indices, null, capacity, size, copy, checkIndicesSorted,
				removeDuplicates);
	}

	public SparseVector(int[] indices, boolean copy, boolean checkIndicesSorted) {
		this(indices, null, copy, checkIndicesSorted, true);
	}

	public SparseVector(int[] indices, boolean copy) {
		this(indices, null, copy, true, true);
	}

	public SparseVector(int[] indices) {
		this(indices, null, true, true, true);
	}

	private static double[] newArrayOfValue(int length, double value) {
		double[] ret = new double[length];
		Arrays.fill(ret, value);
		return ret;
	}

	public boolean isBinary() {
		return values == null;
	}

	public int[] getIndices() {
		return indices;
	}

	public double[] getValues() {
		return values;
	}

	// This is just the number of non-zero entries...
	public int numLocations() {
		return (values == null ? (indices == null ? 0 : indices.length)
				: values.length);
	}

	public int location(int index) {
		if (indices == null)
			return index;
		else
			return Arrays.binarySearch(indices, index);
	}

	public double valueAtLocation(int location) {
		return values == null ? 1.0 : values[location];
	}

	public int indexAtLocation(int location) {
		return indices == null ? location : indices[location];
	}

	/** Sets every present index in the vector to v. */
	public void setAll(double v) {
		for (int i = 0; i < values.length; i++)
			values[i] = v;
	}

	/** Sets the value at the given location. */
	public void setValueAtLocation(int location, double value) {
		values[location] = value;
	}

	/**
	 * Sets the value at the given index.
	 * 
	 * @throws IllegalArgumentException
	 *             If index is not present.
	 */
	public void setValue(int index, double value)
			throws IllegalArgumentException {
		if (indices == null)
			values[index] = value;
		else {
			int loc = location(index);
			if (loc < 0)
				throw new IllegalArgumentException(
						"Can't insert values into a sparse Vector.");
			else
				values[loc] = value;
		}
	}

	/***********************************************************************
	 * VECTOR OPERATIONS
	 ***********************************************************************/

	public double oneNorm() {
		double ret = 0;
		if (values == null)
			return indices.length;
		for (int i = 0; i < values.length; i++)
			ret += values[i];
		return ret;
	}

	public double absNorm() {
		double ret = 0;
		if (values == null)
			return indices.length;
		for (int i = 0; i < values.length; i++)
			ret += Math.abs(values[i]);
		return ret;
	}

	public double twoNorm() {
		double ret = 0;
		if (values == null)
			return Math.sqrt(indices.length);
		for (int i = 0; i < values.length; i++)
			ret += values[i] * values[i];
		return Math.sqrt(ret);
	}

	public double infinityNorm() {
		if (values == null)
			return 1.0;
		double max = Double.NEGATIVE_INFINITY;
		for (int i = 0; i < values.length; i++)
			if (Math.abs(values[i]) > max)
				max = Math.abs(values[i]);
		return max;
	}

	public void print() {
		if (values == null) {
			// binary sparsevector
			for (int i = 0; i < indices.length; i++)
				System.out.println("SparseVector[" + indices[i] + "] = 1.0");
		} else {
			for (int i = 0; i < values.length; i++) {
				int idx = (indices == null) ? i : indices[i];
				System.out.println("SparseVector[" + idx + "] = " + values[i]);
			}
		}
	}

	protected void sortIndices()
	// public void sortIndices () //modified by Limin Yao
	{
		if (indices == null)
			// It's dense, and thus by definition sorted.
			return;
		if (values == null)
			java.util.Arrays.sort(indices);
		else {
			// Just BubbleSort; this is efficient when already mostly sorted.
			// Note that we BubbleSort from the the end forward; this is most
			// efficient
			// when we have added a few additional items to the end of a
			// previously sorted list.
			// We could be much smarter if we remembered the highest index that
			// was already sorted
			for (int i = indices.length - 1; i >= 0; i--) {
				boolean swapped = false;
				for (int j = 0; j < i; j++)
					if (indices[j] > indices[j + 1]) {
						// Swap both indices and values
						int f;
						f = indices[j];
						indices[j] = indices[j + 1];
						indices[j + 1] = f;
						if (values != null) {
							double v;
							v = values[j];
							values[j] = values[j + 1];
							values[j + 1] = v;
						}
						swapped = true;
					}
				if (!swapped)
					break;
			}
		}

		// if (values == null)
		int numDuplicates = 0;
		for (int i = 1; i < indices.length; i++)
			if (indices[i - 1] == indices[i])
				numDuplicates++;

		if (numDuplicates > 0)
			removeDuplicates(numDuplicates);
	}

	// Argument zero is special value meaning that this function should count
	// them.
	protected void removeDuplicates(int numDuplicates) {
		if (numDuplicates == 0)
			for (int i = 1; i < indices.length; i++)
				if (indices[i - 1] == indices[i])
					numDuplicates++;
		if (numDuplicates == 0)
			return;
		int[] newIndices = new int[indices.length - numDuplicates];
		double[] newValues = values == null ? null : new double[indices.length
				- numDuplicates];
		newIndices[0] = indices[0];
		if (values != null)
			newValues[0] = values[0];
		for (int i = 1, j = 1; i < indices.length; i++) {
			if (indices[i] == indices[i - 1]) {
				if (newValues != null)
					newValues[j - 1] += values[i];
			} else {
				newIndices[j] = indices[i];
				if (values != null)
					newValues[j] = values[i];
				j++;
			}
		}
		this.indices = newIndices;
		this.values = newValues;
	}

	public String toString() {

		StringBuffer sb = new StringBuffer();

		for (int i = 0; i < values.length; i++) {
			sb.append((indices == null ? i : indices[i]));
			sb.append("=");
			sb.append(values[i]);
			sb.append("\n");
		}

		return sb.toString();
	}

	public double dotProduct(SparseVector v) {
		double ret = 0;
		for (int i = 0; i < indices.length; i++)
			ret += values[i] * v.values[indices[i]];

		return ret;
	}

	public void plusEqualsSparse(SparseVector v, double factor) {

		int loc1 = 0;
		int loc2 = 0;
		int numLocations1 = numLocations();
		int numLocations2 = v.numLocations();

		while ((loc1 < numLocations1) && (loc2 < numLocations2)) {
			int idx1 = indexAtLocation(loc1);
			int idx2 = v.indexAtLocation(loc2);
			if (idx1 == idx2) {
				values[loc1] += v.valueAtLocation(loc2) * factor;
				++loc1;
				++loc2;
			} else if (idx1 < idx2) {
				++loc1;
			} else {
				// idx2 not present in this. Ignore.
				++loc2;
			}
		}
	}

}
