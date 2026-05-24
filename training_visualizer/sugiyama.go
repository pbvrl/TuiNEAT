//  Defines renderNetworkInSuigyamaLayout() to be used by main

// WARNING: LLM vibe coded, didn't read. Crashes on some networks.

// TODO: Do this module properly

package main

import (
	"fmt"
	"sort"

	"github.com/gdamore/tcell/v3"
)

type NdCoords struct{ X, Y int }

type SugiyamaLayout struct {
	NodeCoords map[int]NdCoords
	EdgePaths  map[[2]int][]NdCoords
	MinX, MinY int
	MaxX, MaxY int
}

type LayerPosition struct{ Layer, Order int }

type LayeredGraph struct {
	NodePosition map[int]LayerPosition
	Segments     map[[2]int]bool
	Dummy        map[int]bool
	Edges        map[[2]int][]int // original edge -> path of node IDs (including dummies)
}

func renderNetworkInSugiyamaLayout(screen tcell.Screen, network Network, availableWidth, availableHeight, x0, y0 int, minWeight, maxWeight, minBias, maxBias, ctrnnMinAlpha, ctrnnMaxAlpha float32, feedforward bool, inputNames, outputNames []string) {
	layout := computeSugiyamaLayout(network, availableWidth, availableHeight, x0, y0)
	renderSugiyamaNetwork(screen, network, layout, minWeight, maxWeight, minBias, maxBias, ctrnnMinAlpha, ctrnnMaxAlpha, feedforward, inputNames, outputNames)
}

func computeSugiyamaLayout(network Network, width, height, x0, y0 int) SugiyamaLayout {
	layout := SugiyamaLayout{
		NodeCoords: make(map[int]NdCoords),
		EdgePaths:  make(map[[2]int][]NdCoords),
	}
	if len(network.ToposrtdNds) == 0 {
		return layout
	}

	// Classify nodes by type
	nodes := make(map[int]bool)
	inputNodes := make(map[int]bool)
	outputNodes := make(map[int]bool)
	for _, nd := range network.ToposrtdNds {
		nodes[nd.Id] = true
		switch nd.Type {
		case 1:
			inputNodes[nd.Id] = true
		case 2:
			outputNodes[nd.Id] = true
		}
	}

	edges := make(map[[2]int]bool)
	for _, c := range network.Cs {
		if c.Enabled {
			edges[[2]int{c.InId, c.OutId}] = true
		}
	}

	// Filter to nodes on input->output paths
	fwd := bfsReachable(inputNodes, edges, true)
	bwd := bfsReachable(outputNodes, edges, false)
	for n := range nodes {
		if !inputNodes[n] && !outputNodes[n] && !(fwd[n] && bwd[n]) {
			delete(nodes, n)
		}
	}
	for e := range edges {
		if !nodes[e[0]] || !nodes[e[1]] {
			delete(edges, e)
		}
	}

	// Layout pipeline
	reversed := removeCycles(nodes, edges)
	positions := assignLayers(nodes, edges)

	// Pin inputs to layer 0, outputs to max layer
	maxLayer := 0
	for _, pos := range positions {
		if pos.Layer > maxLayer {
			maxLayer = pos.Layer
		}
	}
	for n := range inputNodes {
		if nodes[n] {
			positions[n] = LayerPosition{Layer: 0}
		}
	}
	for n := range outputNodes {
		if nodes[n] {
			positions[n] = LayerPosition{Layer: maxLayer}
		}
	}

	lg := createLayeredGraph(nodes, edges, positions)
	reduceCrossings(&lg, 30)

	layers := lg.getLayers()
	numLayers := len(layers)
	maxInLayer := 0
	for _, layer := range layers {
		if len(layer) > maxInLayer {
			maxInLayer = len(layer)
		}
	}
	if numLayers == 0 || maxInLayer == 0 {
		return layout
	}

	usableW, usableH := width-15, height-4 // label margins
	deltaX, deltaY := 12, 3
	if numLayers > 1 {
		deltaX = usableW / (numLayers - 1)
	}
	if maxInLayer > 1 {
		deltaY = usableH / (maxInLayer - 1)
	}
	deltaX = clamp(deltaX, 8, 70)
	deltaY = clamp(deltaY, 2, 5)

	coords := assignCoordinates(&lg, deltaX, deltaY, width, height)
	edgePaths := buildEdgePaths(&lg, coords)

	// Restore reversed edges
	for e := range reversed {
		delete(edges, [2]int{e[1], e[0]})
		edges[e] = true
	}

	// Map to screen coordinates
	for id := range nodes {
		if c, ok := coords[id]; ok {
			layout.NodeCoords[id] = NdCoords{c.X + x0, c.Y + y0}
		}
	}
	for e, path := range edgePaths {
		if !edges[e] {
			continue
		}
		op := make([]NdCoords, len(path))
		for i, p := range path {
			op[i] = NdCoords{p.X + x0, p.Y + y0}
		}
		layout.EdgePaths[e] = op
	}

	layout.MinX, layout.MinY = 1<<30, 1<<30
	layout.MaxX, layout.MaxY = -(1 << 30), -(1 << 30)
	for _, c := range layout.NodeCoords {
		layout.MinX = min(layout.MinX, c.X)
		layout.MaxX = max(layout.MaxX, c.X)
		layout.MinY = min(layout.MinY, c.Y)
		layout.MaxY = max(layout.MaxY, c.Y)
	}
	return layout
}

// bfsReachable finds all nodes reachable from seeds (forward or backward along edges).
func bfsReachable(seeds map[int]bool, edges map[[2]int]bool, forward bool) map[int]bool {
	reached := make(map[int]bool)
	queue := make([]int, 0, len(seeds))
	for n := range seeds {
		reached[n] = true
		queue = append(queue, n)
	}
	for len(queue) > 0 {
		cur := queue[0]
		queue = queue[1:]
		for e := range edges {
			var src, dst int
			if forward {
				src, dst = e[0], e[1]
			} else {
				src, dst = e[1], e[0]
			}
			if src == cur && !reached[dst] {
				reached[dst] = true
				queue = append(queue, dst)
			}
		}
	}
	return reached
}

// removeCycles reverses back-edges to make the graph acyclic.
func removeCycles(nodes map[int]bool, edges map[[2]int]bool) map[[2]int]bool {
	reversed := make(map[[2]int]bool)
	neighbors := buildNeighbors(edges)
	roots := sortedRoots(nodes, edges)

	for {
		cycle := findCycle(roots, neighbors)
		if len(cycle) == 0 {
			break
		}
		e := [2]int{cycle[0], cycle[1]}
		delete(edges, e)
		edges[[2]int{e[1], e[0]}] = true
		reversed[e] = true

		neighbors[e[0]] = removeVal(neighbors[e[0]], e[1])
		neighbors[e[1]] = append(neighbors[e[1]], e[0])
		roots = sortedRoots(nodes, edges)
	}
	return reversed
}

func findCycle(roots []int, neighbors map[int][]int) []int {
	for _, root := range roots {
		if c := findCycleDFS(neighbors, []int{root}); len(c) > 0 {
			return c
		}
	}
	return nil
}

func findCycleDFS(neighbors map[int][]int, path []int) []int {
	cur := path[len(path)-1]
	for _, next := range neighbors[cur] {
		for i, n := range path {
			if n == next {
				return path[i:]
			}
		}
		if c := findCycleDFS(neighbors, append(path, next)); len(c) > 0 {
			return c
		}
	}
	return nil
}

// assignLayers does BFS from roots; sinks get pushed to the max layer.
func assignLayers(nodes map[int]bool, edges map[[2]int]bool) map[int]LayerPosition {
	positions := make(map[int]LayerPosition)
	neighbors := buildNeighbors(edges)
	hasParent := make(map[int]bool)
	hasChild := make(map[int]bool)
	for e := range edges {
		hasParent[e[1]] = true
		hasChild[e[0]] = true
	}

	roots := make([]int, 0)
	for n := range nodes {
		if !hasParent[n] {
			roots = append(roots, n)
		}
	}
	sort.Ints(roots)

	for _, root := range roots {
		positions[root] = LayerPosition{Layer: 0}
		queue := []int{root}
		visited := make(map[int]bool)
		for len(queue) > 0 {
			cur := queue[0]
			queue = queue[1:]
			if visited[cur] {
				continue
			}
			visited[cur] = true
			for _, child := range neighbors[cur] {
				newLayer := positions[cur].Layer + 1
				if newLayer > positions[child].Layer {
					positions[child] = LayerPosition{Layer: newLayer}
				}
				queue = append(queue, child)
			}
		}
	}

	maxLayer := 0
	for _, pos := range positions {
		if pos.Layer > maxLayer {
			maxLayer = pos.Layer
		}
	}
	for n := range nodes {
		if !hasChild[n] && hasParent[n] {
			positions[n] = LayerPosition{Layer: maxLayer}
		}
	}
	return positions
}

// createLayeredGraph inserts dummy nodes for multi-layer edges.
func createLayeredGraph(nodes map[int]bool, edges map[[2]int]bool, positions map[int]LayerPosition) LayeredGraph {
	lg := LayeredGraph{
		NodePosition: make(map[int]LayerPosition),
		Segments:     make(map[[2]int]bool),
		Dummy:        make(map[int]bool),
		Edges:        make(map[[2]int][]int),
	}
	for n, pos := range positions {
		lg.NodePosition[n] = pos
	}

	nextID := 0
	for n := range nodes {
		if n > nextID {
			nextID = n
		}
	}
	nextID++

	for _, e := range sortedEdgeKeys(edges) {
		fromLayer := positions[e[0]].Layer
		toLayer := positions[e[1]].Layer
		path := []int{e[0]}
		for layer := fromLayer + 1; layer < toLayer; layer++ {
			lg.NodePosition[nextID] = LayerPosition{Layer: layer}
			lg.Dummy[nextID] = true
			path = append(path, nextID)
			nextID++
		}
		path = append(path, e[1])
		lg.Edges[e] = path
		for i := 0; i < len(path)-1; i++ {
			lg.Segments[[2]int{path[i], path[i+1]}] = true
		}
	}
	return lg
}

func (lg *LayeredGraph) getLayers() [][]int {
	maxLayer := 0
	for _, pos := range lg.NodePosition {
		if pos.Layer > maxLayer {
			maxLayer = pos.Layer
		}
	}
	layers := make([][]int, maxLayer+1)
	for node, pos := range lg.NodePosition {
		layers[pos.Layer] = append(layers[pos.Layer], node)
	}
	for i := range layers {
		sort.Slice(layers[i], func(a, b int) bool {
			oa, ob := lg.NodePosition[layers[i][a]].Order, lg.NodePosition[layers[i][b]].Order
			if oa != ob {
				return oa < ob
			}
			return layers[i][a] < layers[i][b]
		})
	}
	return layers
}

// reduceCrossings applies median heuristic with forward/backward sweeps.
func reduceCrossings(lg *LayeredGraph, epochs int) {
	layers := lg.getLayers()
	for li, layer := range layers {
		for order, node := range layer {
			pos := lg.NodePosition[node]
			pos.Order = order
			pos.Layer = li
			lg.NodePosition[node] = pos
		}
	}

	bestCrossings := countAllCrossings(lg, layers)
	bestOrder := copyPositions(lg.NodePosition)

	for epoch := 0; epoch < epochs; epoch++ {
		for i := 1; i < len(layers); i++ {
			reorderByMedian(lg, layers, i, true)
		}
		for i := len(layers) - 2; i >= 1; i-- {
			reorderByMedian(lg, layers, i, false)
		}
		crossings := countAllCrossings(lg, layers)
		if crossings < bestCrossings {
			bestCrossings = crossings
			bestOrder = copyPositions(lg.NodePosition)
		}
		if crossings == 0 {
			break
		}
	}
	for node, pos := range bestOrder {
		lg.NodePosition[node] = pos
	}
}

func countAllCrossings(lg *LayeredGraph, layers [][]int) int {
	total := 0
	for i := 0; i < len(layers)-1; i++ {
		total += countCrossingsBetween(lg, layers[i], layers[i+1])
	}
	return total
}

func countCrossingsBetween(lg *LayeredGraph, upper, lower []int) int {
	crossings := 0
	for i := 0; i < len(upper); i++ {
		for j := i + 1; j < len(upper); j++ {
			for _, t1 := range lower {
				if !lg.Segments[[2]int{upper[i], t1}] {
					continue
				}
				for _, t2 := range lower {
					if lg.Segments[[2]int{upper[j], t2}] && lg.NodePosition[t1].Order > lg.NodePosition[t2].Order {
						crossings++
					}
				}
			}
		}
	}
	return crossings
}

func reorderByMedian(lg *LayeredGraph, layers [][]int, layerIdx int, useLeft bool) {
	layer := layers[layerIdx]
	medians := make(map[int]float64)

	for _, node := range layer {
		var npos []int
		if useLeft && layerIdx > 0 {
			for _, ln := range layers[layerIdx-1] {
				if lg.Segments[[2]int{ln, node}] {
					npos = append(npos, lg.NodePosition[ln].Order)
				}
			}
		} else if !useLeft && layerIdx < len(layers)-1 {
			for _, rn := range layers[layerIdx+1] {
				if lg.Segments[[2]int{node, rn}] {
					npos = append(npos, lg.NodePosition[rn].Order)
				}
			}
		}
		if len(npos) > 0 {
			sort.Ints(npos)
			mid := len(npos) / 2
			if len(npos)%2 == 0 {
				medians[node] = float64(npos[mid-1]+npos[mid]) / 2.0
			} else {
				medians[node] = float64(npos[mid])
			}
		} else {
			medians[node] = float64(lg.NodePosition[node].Order)
		}
	}

	sort.Slice(layer, func(i, j int) bool {
		return medians[layer[i]] < medians[layer[j]]
	})
	for order, node := range layer {
		pos := lg.NodePosition[node]
		pos.Order = order
		lg.NodePosition[node] = pos
	}
	layers[layerIdx] = layer
}

func assignCoordinates(lg *LayeredGraph, deltaX, deltaY, availW, availH int) map[int]NdCoords {
	coords := make(map[int]NdCoords)
	for node, pos := range lg.NodePosition {
		coords[node] = NdCoords{pos.Layer * deltaX, pos.Order * deltaY}
	}

	// Pull nodes toward neighbor averages
	layers := lg.getLayers()
	for iter := 0; iter < 10; iter++ {
		for li := 1; li < len(layers); li++ {
			refineLayerCoords(lg, layers, li, coords, deltaY, true)
		}
		for li := len(layers) - 2; li >= 0; li-- {
			refineLayerCoords(lg, layers, li, coords, deltaY, false)
		}
	}

	// Center in available space
	if len(coords) == 0 {
		return coords
	}
	minX, maxX, minY, maxY := 1<<30, -(1 << 30), 1<<30, -(1 << 30)
	for _, c := range coords {
		minX = min(minX, c.X)
		maxX = max(maxX, c.X)
		minY = min(minY, c.Y)
		maxY = max(maxY, c.Y)
	}
	topMargin, rightMargin := 2, 15
	usableW := availW - rightMargin
	usableH := availH - topMargin
	offsetX := max((usableW-(maxX-minX))/2-minX, -minX)
	offsetY := max(topMargin+(usableH-(maxY-minY))/2-minY, topMargin-minY)
	for node, c := range coords {
		coords[node] = NdCoords{c.X + offsetX, c.Y + offsetY}
	}
	return coords
}

func refineLayerCoords(lg *LayeredGraph, layers [][]int, layerIdx int, coords map[int]NdCoords, deltaY int, useLeft bool) {
	layer := layers[layerIdx]
	if len(layer) == 0 || layerIdx == 0 {
		return
	}

	// Target Y from neighbor averages
	targets := make(map[int]float64)
	for _, node := range layer {
		var ys []int
		if useLeft && layerIdx > 0 {
			for _, ln := range layers[layerIdx-1] {
				if lg.Segments[[2]int{ln, node}] {
					ys = append(ys, coords[ln].Y)
				}
			}
		} else if !useLeft && layerIdx < len(layers)-1 {
			for _, rn := range layers[layerIdx+1] {
				if lg.Segments[[2]int{node, rn}] {
					ys = append(ys, coords[rn].Y)
				}
			}
		}
		if len(ys) > 0 {
			sum := 0
			for _, y := range ys {
				sum += y
			}
			targets[node] = float64(sum) / float64(len(ys))
		} else {
			targets[node] = float64(coords[node].Y)
		}
	}

	// Assign Y with minimum spacing
	type nt struct {
		node int
		tgt  float64
		orig int
	}
	nts := make([]nt, len(layer))
	for i, n := range layer {
		nts[i] = nt{n, targets[n], i}
	}
	sort.Slice(nts, func(i, j int) bool {
		if nts[i].tgt != nts[j].tgt {
			return nts[i].tgt < nts[j].tgt
		}
		return nts[i].orig < nts[j].orig
	})

	curY := max(int(nts[0].tgt), 0)
	for i, n := range nts {
		y := max(int(n.tgt), curY)
		if i > 0 && y > curY+deltaY*2 {
			y = curY + deltaY
		}
		c := coords[n.node]
		c.Y = y
		coords[n.node] = c
		curY = y + deltaY
	}
}

func buildEdgePaths(lg *LayeredGraph, coords map[int]NdCoords) map[[2]int][]NdCoords {
	paths := make(map[[2]int][]NdCoords)
	for e, nodePath := range lg.Edges {
		ep := make([]NdCoords, len(nodePath))
		for i, node := range nodePath {
			ep[i] = coords[node]
		}
		paths[e] = ep
	}
	return paths
}

// Rendering

func renderSugiyamaNetwork(screen tcell.Screen, network Network, layout SugiyamaLayout,
	minWeight, maxWeight, minBias, maxBias, ctrnnMinAlpha, ctrnnMaxAlpha float32, feedforward bool, inputNames, outputNames []string) {

	nodeMap := make(map[int]Nd)
	var inputIDs, outputIDs []int
	for _, nd := range network.ToposrtdNds {
		nodeMap[nd.Id] = nd
		switch nd.Type {
		case 1:
			inputIDs = append(inputIDs, nd.Id)
		case 2:
			outputIDs = append(outputIDs, nd.Id)
		}
	}
	sort.Ints(inputIDs)
	sort.Ints(outputIDs)

	inputNameMap := buildNameMap(inputIDs, inputNames)
	outputNameMap := buildNameMap(outputIDs, outputNames)

	weightMap := make(map[[2]int]float32)
	for _, c := range network.Cs {
		if c.Enabled {
			weightMap[[2]int{c.InId, c.OutId}] = c.Weight
		}
	}

	// Edges first (behind nodes)
	for edgeIdx, ek := range sortedEdgeKeys(layout.EdgePaths) {
		path := layout.EdgePaths[ek]
		weight := weightMap[ek]
		color, err := weightColorGradient.pick(weight, minWeight, maxWeight)
		if err != nil {
			color = tcell.ColorWhite
		}
		style := tcell.StyleDefault.Foreground(color)
		drawEdge(screen, path, style, edgeIdx%2 == 1)
	}

	// Nodes on top
	sortedNodes := make([]int, 0, len(layout.NodeCoords))
	for id := range layout.NodeCoords {
		sortedNodes = append(sortedNodes, id)
	}
	sort.Ints(sortedNodes)

	for _, id := range sortedNodes {
		coord := layout.NodeCoords[id]
		nd, ok := nodeMap[id]
		if !ok {
			continue
		}

		actColor := activationCodes[nd.Activation].Color
		aggColor := aggregationCodes[nd.Aggregation].Color
		idStr := fmt.Sprintf("%d", nd.Id)
		for i, r := range idStr {
			screen.SetContent(coord.X+i, coord.Y, r, nil, tcell.StyleDefault.Foreground(actColor).Background(aggColor))
		}

		if nd.Type != 1 {
			if biasColor, err := weightColorGradient.pick(nd.Bias, minBias, maxBias); err == nil {
				screen.SetContent(coord.X, coord.Y-1, '&', nil, tcell.StyleDefault.Foreground(biasColor))
			}
			if !feedforward {
				if alphaColor, err := weightColorGradient.pick(nd.CtrnnAlpha, ctrnnMinAlpha, ctrnnMaxAlpha); err == nil {
					screen.SetContent(coord.X, coord.Y-2, '%', nil, tcell.StyleDefault.Foreground(alphaColor))
				}
			}
		}

		if nd.Type == 1 {
			if name, ok := inputNameMap[id]; ok {
				drawLabel(screen, coord.X-len(name)-1, coord.Y, name)
			}
		}
		if nd.Type == 2 {
			if name, ok := outputNameMap[id]; ok {
				drawLabel(screen, coord.X+len(idStr)+1, coord.Y, name)
			}
		}
	}
}

func drawLabel(screen tcell.Screen, x, y int, text string) {
	style := tcell.StyleDefault.Foreground(tcell.NewRGBColor(128, 128, 128))
	for i, r := range text {
		screen.SetContent(x+i, y, r, nil, style)
	}
}

func drawEdge(screen tcell.Screen, path []NdCoords, style tcell.Style, leanUp bool) {
	if len(path) < 2 {
		return
	}
	for i := 0; i < len(path)-1; i++ {
		x1, y1 := path[i].X, path[i].Y
		if i == 0 {
			x1 += 2
		}
		drawLine(screen, x1, y1, path[i+1].X, path[i+1].Y, style)
	}
}

func drawLine(screen tcell.Screen, x1, y1, x2, y2 int, style tcell.Style) {
	dx, dy := x2-x1, y2-y1
	if dx == 0 && dy == 0 {
		return
	}
	sx, sy := sign(dx), sign(dy)
	adx, ady := abs(dx), abs(dy)

	if dy == 0 {
		for x := x1; x != x2+sx; x += sx {
			screen.SetContent(x, y1, '─', nil, style)
		}
		return
	}
	if dx == 0 {
		for y := y1; y != y2+sy; y += sy {
			screen.SetContent(x1, y, '│', nil, style)
		}
		return
	}

	if adx >= ady {
		step := float64(adx) / float64(ady+1)
		x := x1
		for row := 0; row <= ady; row++ {
			y := y1 + row*sy
			endX := x2
			if row < ady {
				endX = x1 + int(float64(row+1)*step)*sx
			}
			for ; x != endX+sx; x += sx {
				screen.SetContent(x, y, '─', nil, style)
			}
			if row < ady {
				corner := '╮'
				if sy < 0 {
					corner = '╯'
				}
				screen.SetContent(x-sx, y, corner, nil, style)
			}
		}
	} else {
		step := float64(ady) / float64(adx+1)
		y := y1
		for col := 0; col <= adx; col++ {
			x := x1 + col*sx
			endY := y2
			if col < adx {
				endY = y1 + int(float64(col+1)*step)*sy
			}
			for ; y != endY+sy; y += sy {
				screen.SetContent(x, y, '│', nil, style)
			}
			if col < adx {
				var corner rune
				if sx > 0 {
					if sy > 0 {
						corner = '╰'
					} else {
						corner = '╭'
					}
				} else {
					if sy > 0 {
						corner = '╯'
					} else {
						corner = '╮'
					}
				}
				screen.SetContent(x, y-sy, corner, nil, style)
			}
		}
	}
}

// Helpers

func buildNeighbors(edges map[[2]int]bool) map[int][]int {
	neighbors := make(map[int][]int)
	for e := range edges {
		neighbors[e[0]] = append(neighbors[e[0]], e[1])
	}
	for n := range neighbors {
		sort.Ints(neighbors[n])
	}
	return neighbors
}

func sortedRoots(nodes map[int]bool, edges map[[2]int]bool) []int {
	hasParent := make(map[int]bool)
	for e := range edges {
		hasParent[e[1]] = true
	}
	var roots []int
	for n := range nodes {
		if !hasParent[n] {
			roots = append(roots, n)
		}
	}
	sort.Ints(roots)
	return roots
}

func sortedEdgeKeys[V any](m map[[2]int]V) [][2]int {
	keys := make([][2]int, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if keys[i][0] != keys[j][0] {
			return keys[i][0] < keys[j][0]
		}
		return keys[i][1] < keys[j][1]
	})
	return keys
}

func copyPositions(pos map[int]LayerPosition) map[int]LayerPosition {
	cp := make(map[int]LayerPosition, len(pos))
	for k, v := range pos {
		cp[k] = v
	}
	return cp
}

func buildNameMap(ids []int, names []string) map[int]string {
	m := make(map[int]string)
	for i, id := range ids {
		if i < len(names) {
			m[id] = names[i]
		}
	}
	return m
}

func removeVal(s []int, val int) []int {
	for i, v := range s {
		if v == val {
			return append(s[:i], s[i+1:]...)
		}
	}
	return s
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func sign(x int) int {
	if x > 0 {
		return 1
	}
	if x < 0 {
		return -1
	}
	return 0
}

func clamp(v, lo, hi int) int {
	if v < lo {
		return lo
	}
	if v > hi {
		return hi
	}
	return v
}
