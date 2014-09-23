package edu.pengli.nlp.platform.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Set;

public class RankMap {
	
//should import key indexed counting algorithm at P705 
	
// should import comparator method, I found it at Jan 23, 2013 in Samsung.
	
	// this method has the time complexity of O(n).
	public static LinkedHashMap sortHashMapByValues(
			HashMap passedMap, boolean ascending) {
		List mapKeys = new ArrayList(passedMap.keySet());
		Iterator keyIt = mapKeys.iterator();
		HashMap newValToKeysMap = new HashMap();
		LinkedList tmpVals = new LinkedList();
		while (keyIt.hasNext()) {
			Object key = keyIt.next();
			Object val = passedMap.get(key);
			LinkedList keyList = null;
			if (!tmpVals.contains(val)) {
				tmpVals.add(val);
				keyList = new LinkedList();
				keyList.add(key);

			} else {
				keyList = (LinkedList) newValToKeysMap.get(val);
				keyList.add(key);

			}
			newValToKeysMap.put(val, keyList);

		}
		
		List mapValues = new ArrayList(newValToKeysMap.keySet());
		Collections.sort(mapValues);
		LinkedHashMap returnedMap = new LinkedHashMap();
		if (!ascending)
			Collections.reverse(mapValues);
		Iterator valIt = mapValues.iterator();
		while(valIt.hasNext()){
			Object val = valIt.next();
			LinkedList keys = (LinkedList) newValToKeysMap.get(val);
			for(int i=0 ; i<keys.size(); i++){
				returnedMap.put(keys.get(i), val);
			}
			
		}
		return returnedMap;
	}

	public static LinkedHashMap sortElementsInListByCount(List sList,
			boolean ascending) {

		LinkedHashMap map = new LinkedHashMap();
		HashSet itemSet = new HashSet();
		itemSet.addAll(sList);

		Iterator keyIt = itemSet.iterator();
		while (keyIt.hasNext()) {
			Object key = keyIt.next();
			Iterator listIter = sList.iterator();
			int count = 0;
			while (listIter.hasNext()) {
				Object item = listIter.next();
				if (key.equals(item)) {
					map.put(key, ++count);
				}
			}
		}

		return sortHashMapByValues(map, ascending);

	}
	public static LinkedHashMap sortElementsInLargeListByCount(List sList, boolean ascending){
		
		LinkedHashMap map = new LinkedHashMap();
		HashSet itemSet = new HashSet();
		itemSet.addAll(sList);

		Iterator keyIt = itemSet.iterator();
		while (keyIt.hasNext()) {
			Object key = keyIt.next();
			Iterator listIter = sList.iterator();
			int count = 0;
			while (listIter.hasNext()) {
				Object item = listIter.next();
				if (key.equals(item)) {
					map.put(key, ++count);
				}
			}
		}
		return sortHashMapByValues(map, ascending);
	}
	//test
	public static void main(String[] args){
		
		HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
		map.put(1, 5);
		map.put(2, 6);
		map.put(3, 4);
		map.put(4, 7);
		
		LinkedHashMap ranked = sortHashMapByValues(map, true);
		Set<Integer> keys = ranked.keySet();
		
		for(Integer i : keys){
			System.out.println(i+"::"+map.get(i));
		}
		
	}
}
