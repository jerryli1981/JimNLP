package edu.pengli.nlp.platform.algorithms.miscellaneous;


public class LongestCommonSubstring {

	public static double getSim(String s1, String s2) {
		String lcs = LongestCommonSubstring.lcs(s1, s2);

		double sim = (double) lcs.length()
				/ (double) Math.min(s1.length(), s2.length());
		return sim;
	}

	public static String lcp(String s, String t) {
		int n = Math.min(s.length(), t.length());
		for (int i = 0; i < n; i++) {
			if (s.charAt(i) != t.charAt(i))
				return s.substring(0, i);
		}
		return s.substring(0, n);
	}

	// return the longest common substring that appears in both s and t
	public static String lcs(String s, String t) {
		SuffixArray suffix1 = new SuffixArray(s);
		SuffixArray suffix2 = new SuffixArray(t);

		// find longest common substring by comparing sorted suffixes
		String lcs = "";
		int i = 0, j = 0;
		while (i < s.length() && j < t.length()) {
			String a = suffix1.ith(i);
			String b = suffix2.ith(j);
			String x = lcp(a, b);
			if (x.length() > lcs.length())
				lcs = x;
			if (a.compareTo(b) < 0)
				i++;
			else
				j++;
		}
		return lcs;
	}

	public static String LCS(String first, String second) {

		String tmp = "";
		String max = "";

		for (int i = 0; i < first.length(); i++) {
			for (int j = 0; j < second.length(); j++) {
				for (int k = 1; (k + i) <= first.length()
						&& (k + j) <= second.length(); k++) {

					if (first.substring(i, k + i).equals(
							second.substring(j, k + j))) {
						tmp = first.substring(i, k + i);
					} else {
						if (tmp.length() > max.length())
							max = tmp;
						tmp = "";
					}
				}
				if (tmp.length() > max.length())
					max = tmp;
				tmp = "";
			}
		}

		return max;

	}

}

class SuffixArray {
	private final String s; // the original string
	private final int[] index; // index[i] = ith suffix

	// private final int[] lcp; // lcp[i] = longest common prefix of the ith and
	// i-1st suffixes

	public SuffixArray(String s) {
		this.s = s;
		int N = s.length();
		String[] suffixes = new String[N];
		for (int i = 0; i < N; i++)
			suffixes[i] = s.substring(i);
		index = Merge.indexSort(suffixes);
	}

	// index of ith sorted suffix
	public int index(int i) {
		return index[i];
	}

	// ith sorted suffix
	public String ith(int i) {
		return s.substring(index[i]);
	}

}

class Merge {

	// stably merge a[lo .. mid] with a[mid+1 .. hi] using aux[lo .. hi]
	private static void merge(Comparable[] a, Comparable[] aux, int lo,
			int mid, int hi) {

		// precondition: a[lo .. mid] and a[mid+1 .. hi] are sorted subarrays
		assert isSorted(a, lo, mid);
		assert isSorted(a, mid + 1, hi);

		// copy to aux[]
		for (int k = lo; k <= hi; k++) {
			aux[k] = a[k];
		}

		// merge back to a[]
		int i = lo, j = mid + 1;
		for (int k = lo; k <= hi; k++) {
			if (i > mid)
				a[k] = aux[j++];
			else if (j > hi)
				a[k] = aux[i++];
			else if (less(aux[j], aux[i]))
				a[k] = aux[j++];
			else
				a[k] = aux[i++];
		}

		// postcondition: a[lo .. hi] is sorted
		assert isSorted(a, lo, hi);
	}

	// mergesort a[lo..hi] using auxiliary array aux[lo..hi]
	private static void sort(Comparable[] a, Comparable[] aux, int lo, int hi) {
		if (hi <= lo)
			return;
		int mid = lo + (hi - lo) / 2;
		sort(a, aux, lo, mid);
		sort(a, aux, mid + 1, hi);
		merge(a, aux, lo, mid, hi);
	}

	public static void sort(Comparable[] a) {
		Comparable[] aux = new Comparable[a.length];
		sort(a, aux, 0, a.length - 1);
		assert isSorted(a);
	}

	/***********************************************************************
	 * Helper sorting functions
	 ***********************************************************************/

	// is v < w ?
	private static boolean less(Comparable v, Comparable w) {
		return (v.compareTo(w) < 0);
	}

	// exchange a[i] and a[j]
	private static void exch(Object[] a, int i, int j) {
		Object swap = a[i];
		a[i] = a[j];
		a[j] = swap;
	}

	/***********************************************************************
	 * Check if array is sorted - useful for debugging
	 ***********************************************************************/
	private static boolean isSorted(Comparable[] a) {
		return isSorted(a, 0, a.length - 1);
	}

	private static boolean isSorted(Comparable[] a, int lo, int hi) {
		for (int i = lo + 1; i <= hi; i++)
			if (less(a[i], a[i - 1]))
				return false;
		return true;
	}

	/***********************************************************************
	 * Index mergesort
	 ***********************************************************************/
	// stably merge a[lo .. mid] with a[mid+1 .. hi] using aux[lo .. hi]
	private static void merge(Comparable[] a, int[] index, int[] aux, int lo,
			int mid, int hi) {

		// copy to aux[]
		for (int k = lo; k <= hi; k++) {
			aux[k] = index[k];
		}

		// merge back to a[]
		int i = lo, j = mid + 1;
		for (int k = lo; k <= hi; k++) {
			if (i > mid)
				index[k] = aux[j++];
			else if (j > hi)
				index[k] = aux[i++];
			else if (less(a[aux[j]], a[aux[i]]))
				index[k] = aux[j++];
			else
				index[k] = aux[i++];
		}
	}

	// return a permutation that gives the elements in a[] in ascending order
	// do not change the original array a[]
	public static int[] indexSort(Comparable[] a) {
		int N = a.length;
		int[] index = new int[N];
		for (int i = 0; i < N; i++)
			index[i] = i;

		int[] aux = new int[N];
		sort(a, index, aux, 0, N - 1);
		return index;
	}

	// mergesort a[lo..hi] using auxiliary array aux[lo..hi]
	private static void sort(Comparable[] a, int[] index, int[] aux, int lo,
			int hi) {
		if (hi <= lo)
			return;
		int mid = lo + (hi - lo) / 2;
		sort(a, index, aux, lo, mid);
		sort(a, index, aux, mid + 1, hi);
		merge(a, index, aux, lo, mid, hi);
	}

	// test client
	public static void main(String[] args) {

		// generate array of N random reals between 0 and 1
		int N = Integer.parseInt(args[0]);
		Double[] a = new Double[N];
		for (int i = 0; i < N; i++) {
			a[i] = Math.random();
		}

		// sort the array
		sort(a);

		// display results
		for (int i = 0; i < N; i++) {
			System.out.println(a[i]);
		}
		System.out.println("isSorted = " + isSorted(a));
	}
}
