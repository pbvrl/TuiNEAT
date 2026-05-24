package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"math"
	"net"
	"sort"
	"sync"
	"time"

	"github.com/gdamore/tcell/v3"
	"google.golang.org/grpc"

	pb "training_visualizer/grpc_api"
)

type updateData struct {
	data             *pb.TrainingData
	currentHparams   VizTunableHparams
	requestedHparams VizTunableHparams
}
type Server struct {
	mu                    sync.Mutex
	screen                tcell.Screen
	speciesHistory        [][]Species
	maxFitnessHistory     []float32
	genTimestamps         []time.Time
	gen                   int
	modifiedStagnationAge int32
	updatesChannel        chan updateData
	currentHparams        VizTunableHparams
	requestedHparams      VizTunableHparams
	pendingHparams        *VizTunableHparams
	vizNonTunableHparams  VizNonTunableHparams
	selectedHparamIdx     int
	isSugiyamaToggled     bool
	selectedSpeciesIdx    int
	selectedNetworkIdx    int
	isFrozenToggled       bool
	genSnapshot           genSnapshot
	taskName              string
	port                  string
	inputNames            []string
	outputNames           []string
	perfRelatedParams     PerformanceRelatedParams
	pb.TrainingVisualizerServer
}

type Network struct {
	Cs          []C
	ToposrtdNds []Nd
	Fitness     float32
	SpeciesId   int
}
type C struct {
	InId    int
	OutId   int
	Enabled bool
	Weight  float32
}
type Nd struct {
	Id          int
	Type        int
	Aggregation int
	Activation  int
	Bias        float32
	CtrnnAlpha  float32
}
type Species struct {
	Id              int
	Quota           int
	MemberCount     int
	MinFitness      float32
	MaxFitness      float32
	CompatThreshold float32
	Stagnation      int
}
type genSnapshot struct {
	topNetworksBySpeciesId      map[int][]Network
	speciesIdsSortedByChampions []int
	species                     []Species
	prevGenSpecies              []Species
	vizMinWeight                float32
	vizMaxWeight                float32
	vizMinBias                  float32
	vizMaxBias                  float32
	vizCtrnnMinAlpha            float32
	vizCtrnnMaxAlpha            float32
	popAvgFitness               float32
	intervalSincePrev           float64
}
type NdPlotData struct {
	X  int
	Y  int
	Nd Nd
}
type PerformanceRelatedParams struct {
	GpuEnabled      bool
	PopSize         int
	MaxRolloutSteps int
	RolloutRepeats  int
	MaxNodes        int
	MaxConns        int
}

type VizTunableHparams struct {
	AddOneConnProb                    float32
	AddOneNodeProb                    float32
	EraseConnProb                     float32
	PerturbWghtStdev                  float32
	ReplaceWghtStdev                  float32
	PerturbWghtProb                   float32
	ReplaceWghtProb                   float32
	PerturbBiasStdev                  float32
	ReplaceBiasStdev                  float32
	PerturbBiasProb                   float32
	ReplaceBiasProb                   float32
	PerturbAlphaStdev                 float32
	PerturbAlphaProb                  float32
	ReplaceAlphaProb                  float32
	ChangeActProb                     float32
	ChangeAggProb                     float32
	DefaultOutputActivation           bool
	DisableCProb                      float32
	ReenableOneCProb                  float32
	CrossoverRate                     float32
	EnabledRecessivenessProb          float32
	AvgWeightsProb                    float32
	InterspeciesMatingRatio           float32
	IntraspeciesParenthoodEligibility float32
	SpeciesFitnessPowerscaling        float32
	CompatExcessCoeff                 float32
	CompatDisjointCoeff               float32
	CompatWghtCoeff                   float32
	CompatEnabledCoeff                float32
	CompatAggCoeff                    float32
	CompatActCoeff                    float32
	CompatBiasCoeff                   float32
	CompatAlphaCoeff                  float32
	BaseCompatThreshold               float32
	DynamicThresholds                 bool
	DynamicThresholdRatio             float32
	DynamicThresholdLearningRate      float32
	BaseStagnationAge                 float32
	StagnationAgePowerlawGrowth       float32
	NStagnationExemptTopSpecies       float32
}

const (
	hpFloat int = 0
	hpBool  int = 1
)

type hparamMetadata struct {
	Description string
	Kind        int
	Step        float32                             // float only
	FloatPtr    func(h *VizTunableHparams) *float32 // float only
	BoolPtr     func(h *VizTunableHparams) *bool    // bool only
}

var hparamsMetadata = []hparamMetadata{
	{Description: "Add 1 c p.", Kind: hpFloat, Step: 0.001, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.AddOneConnProb }},
	{Description: "Add 1 nd p.", Kind: hpFloat, Step: 0.001, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.AddOneNodeProb }},
	{Description: "Change act. p.", Kind: hpFloat, Step: 0.001, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ChangeActProb }},
	{Description: "Change agg. p.", Kind: hpFloat, Step: 0.001, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ChangeAggProb }},
	{Description: "Default outpt act.", Kind: hpBool, BoolPtr: func(h *VizTunableHparams) *bool { return &h.DefaultOutputActivation }},
	{Description: "Perturb wght stdev", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.PerturbWghtStdev }},
	{Description: "Replace wght stdev", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ReplaceWghtStdev }},
	{Description: "Perturb wght p.", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.PerturbWghtProb }},
	{Description: "Replace wght p.", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ReplaceWghtProb }},
	{Description: "Perturb bias stdev", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.PerturbBiasStdev }},
	{Description: "Replace bias stdev", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ReplaceBiasStdev }},
	{Description: "Perturb bias p.", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.PerturbBiasProb }},
	{Description: "Replace bias p.", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ReplaceBiasProb }},
	{Description: "Perturb alph stdev", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.PerturbAlphaStdev }},
	{Description: "Perturb alpha p.", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.PerturbAlphaProb }},
	{Description: "Replace alpha p.", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ReplaceAlphaProb }},
	{Description: "Disable c. p.", Kind: hpFloat, Step: 0.001, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.DisableCProb }},
	{Description: "Reenable 1 c. p.", Kind: hpFloat, Step: 0.001, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.ReenableOneCProb }},
	{Description: "Erase c p.", Kind: hpFloat, Step: 0.001, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.EraseConnProb }},
	{Description: "Crossover rate", Kind: hpFloat, Step: 0.05, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CrossoverRate }},
	{Description: "Enabled recssv. p.", Kind: hpFloat, Step: 0.05, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.EnabledRecessivenessProb }},
	{Description: "Avg wghts p.", Kind: hpFloat, Step: 0.05, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.AvgWeightsProb }},
	{Description: "Intraspcs prnthood", Kind: hpFloat, Step: 0.05, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.IntraspeciesParenthoodEligibility }},
	{Description: "Interspcs mating  ", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.InterspeciesMatingRatio }},
	{Description: "Sp. fit. pwrscalng", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.SpeciesFitnessPowerscaling }},
	{Description: "Cmptb. excess cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatExcessCoeff }},
	{Description: "Cmptb. disjnt cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatDisjointCoeff }},
	{Description: "Cmptb. wght cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatWghtCoeff }},
	{Description: "Cmptb. enabld cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatEnabledCoeff }},
	{Description: "Cmptb. agg cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatAggCoeff }},
	{Description: "Cmptb. act cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatActCoeff }},
	{Description: "Cmptb. bias cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatBiasCoeff }},
	{Description: "Cmptb. alpha cff.", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.CompatAlphaCoeff }},
	{Description: "Base cmptb. thrhld", Kind: hpFloat, Step: 0.1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.BaseCompatThreshold }},
	{Description: "Dyn. thresholds", Kind: hpBool, BoolPtr: func(h *VizTunableHparams) *bool { return &h.DynamicThresholds }},
	{Description: "Dyn. thrhld ratio", Kind: hpFloat, Step: 0.05, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.DynamicThresholdRatio }},
	{Description: "Dyn. thrhld l.r.", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.DynamicThresholdLearningRate }},
	{Description: "Base stagntion age", Kind: hpFloat, Step: 1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.BaseStagnationAge }},
	{Description: "Stg pwerlaw growth", Kind: hpFloat, Step: 0.01, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.StagnationAgePowerlawGrowth }},
	{Description: "N. stg exempt sp.", Kind: hpFloat, Step: 1, FloatPtr: func(h *VizTunableHparams) *float32 { return &h.NStagnationExemptTopSpecies }},
}

var pendingHparamColor = tcell.NewRGBColor(255, 165, 0)
var requestedHparamColor = tcell.NewRGBColor(0, 200, 200)

type VizNonTunableHparams struct {
	PopSize                           int32
	Feedforward                       bool
	CtrnnIntegrationSteps             int32
	IntraspeciesUnchangedFrontrunners int32
	NormalizeObs                      bool
}

var aggregationCodes = map[int]struct {
	Description string
	Color       tcell.Color
}{
	-1: {"input node", tcell.ColorDefault},
	1:  {"sum", tcell.ColorDefault},
	2:  {"maxabs", tcell.NewRGBColor(100, 100, 100)},
}

var activationCodes = map[int]struct {
	Description string
	Color       tcell.Color
}{
	-1: {"input node", tcell.NewRGBColor(255, 255, 255)},
	0:  {"identity", tcell.NewRGBColor(255, 255, 255)},
	1:  {"tanh", tcell.NewRGBColor(0, 0, 0)},
	2:  {"mish", tcell.NewRGBColor(255, 0, 0)},
	3:  {"sine", tcell.NewRGBColor(0, 200, 200)},
	4:  {"abs", tcell.NewRGBColor(255, 255, 0)},
}

type ColorGradient []tcell.Color

var speciesColorGradient = ColorGradient{
	tcell.NewRGBColor(95, 0, 0),
	tcell.NewRGBColor(95, 0, 255),
	tcell.NewRGBColor(95, 135, 0),
	tcell.NewRGBColor(95, 175, 0),
	tcell.NewRGBColor(95, 255, 0),
	tcell.NewRGBColor(135, 0, 135),
	tcell.NewRGBColor(175, 95, 0),
	tcell.NewRGBColor(175, 175, 0),
	tcell.NewRGBColor(215, 0, 95),
	tcell.NewRGBColor(215, 135, 0),
	tcell.NewRGBColor(255, 215, 0),
	tcell.NewRGBColor(18, 18, 18),
}
var weightColorGradient = ColorGradient{
	tcell.NewRGBColor(255, 0, 255),
	tcell.NewRGBColor(255, 105, 255),
	tcell.NewRGBColor(255, 154, 255),
	tcell.NewRGBColor(255, 196, 255),
	tcell.NewRGBColor(255, 236, 255),
	tcell.NewRGBColor(239, 255, 233),
	tcell.NewRGBColor(206, 255, 188),
	tcell.NewRGBColor(167, 255, 143),
	tcell.NewRGBColor(118, 255, 93),
	tcell.NewRGBColor(0, 255, 0),
}

func (g ColorGradient) pick(value float32, min float32, max float32) (tcell.Color, error) {
	if (value < min) || (value > max) {
		return tcell.NewRGBColor(255, 255, 255), errors.New("To pick a color from the gradient the value must be between min and max")
	}
	if min >= max {
		return g[len(g)/2], nil
	}
	absMax := float32(math.Max(math.Abs(float64(max)), math.Abs(float64(min))))
	if absMax == 0 {
		return g[len(g)/2], nil
	}
	t := (value + absMax) / (2 * absMax)
	index := int(t * float32(len(g)-1))
	return g[index], nil
}

func main() {
	port := flag.String("port", "50051", "Port to run this visualization server on")
	flag.Parse()
	lis, err := net.Listen("tcp", ":"+*port)
	if err != nil {
		log.Fatalf("failed to listen on port %s: %v", *port, err)
	}
	screen, _ := tcell.NewScreen()
	screen.Init()
	defer screen.Fini()
	server := &Server{
		screen:         screen,
		updatesChannel: make(chan updateData, 1),
		port:           *port,
	}
	go server.runUpdatesLoop()
	go server.runEventLoop()
	s := grpc.NewServer()
	pb.RegisterTrainingVisualizerServer(s, server)
	fmt.Printf("gRPC server is running on port: %s\n", *port)
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

func (s *Server) SyncVisualization(ctx context.Context, data *pb.TrainingData) (*pb.HparamsRequest, error) {
	currentHparams := parseVizTunableHparams(data.VizHparams.VizTunableHparams)
	requestHparams := currentHparams
	s.mu.Lock()
	if data.Gen == 0 {
		s.pendingHparams = &currentHparams

	} else if s.pendingHparams != nil {
		requestHparams = *s.pendingHparams
	}
	hparamsRequest := serializeVizTunableHparams(requestHparams)
	s.mu.Unlock()
	update := updateData{data: data, currentHparams: currentHparams, requestedHparams: requestHparams}
	s.updatesChannel <- update
	return &pb.HparamsRequest{VizTunableHparams: hparamsRequest}, nil
}

func (s *Server) runUpdatesLoop() {
	for update := range s.updatesChannel {
		data := update.data
		networks := parseNetworks(data.Networks)
		species := parseSpecies(data.Species)
		perfRelatedParams := parsePerformanceRelatedParams(data.PerfRelatedParams)
		vizNonTunableHparams := parseVizNonTunableHparams(data.VizHparams)
		vizMinWeight, vizMaxWeight := data.MinWeight, data.MaxWeight
		vizMinBias, vizMaxBias := data.MinBias, data.MaxBias
		vizCtrnnMinAlpha, vizCtrnnMaxAlpha := data.CtrnnMinAlpha, data.CtrnnMaxAlpha

		s.mu.Lock()
		if int(data.Gen) != s.gen+1 || int(data.PerfRelatedParams.PopSize) != perfRelatedParams.PopSize {
			s.pendingHparams = nil
			s.speciesHistory = nil
			s.maxFitnessHistory = nil
			s.genTimestamps = nil
			s.inputNames = data.InputNames
			s.outputNames = data.OutputNames
			s.taskName = data.TaskName
			s.perfRelatedParams = perfRelatedParams
		}
		s.modifiedStagnationAge = int32(data.ModifiedStagnationAge)
		s.vizNonTunableHparams = vizNonTunableHparams
		s.gen = int(data.Gen)
		s.updateSpeciesHistory(species)
		s.updateMaxFitnessHistory(species)
		s.updateGenTimestamps()
		s.currentHparams = update.currentHparams
		s.requestedHparams = update.requestedHparams
		if s.isFrozenToggled {
			s.mu.Unlock()
			continue
		}
		s.selectedNetworkIdx = 0
		s.updateSnapshot(networks, species, s.speciesHistory, vizMinWeight, vizMaxWeight, vizMinBias, vizMaxBias, vizCtrnnMinAlpha, vizCtrnnMaxAlpha, float32(data.PopAvgFitness), s.genTimestamps)

		s.render()

		s.mu.Unlock()
	}
}

func (s *Server) clearArea(x0 int, x1 int, y0 int, y1 int) {
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			s.screen.SetContent(x, y, ' ', nil, tcell.StyleDefault)
		}
	}
}

func (s *Server) render() {
	width, height := s.screen.Size()
	s.screen.Clear()
	renderHeader(s.screen, s.gen, s.genSnapshot.popAvgFitness, s.taskName, 0, 0)
	renderSpecies(s.screen, s.genSnapshot.species, s.genSnapshot.prevGenSpecies, s.modifiedStagnationAge, 0, 2)
	renderSpeciesHistory(s.screen, s.speciesHistory, s.perfRelatedParams.PopSize, 15, 71, 0)
	renderActivationsLegend(s.screen, 111, 0)
	renderAggregationsLegend(s.screen, 111, 7)
	renderGradientLegends(s.screen, s.genSnapshot.vizMinWeight, s.genSnapshot.vizMaxWeight, s.genSnapshot.vizMinBias, s.genSnapshot.vizMaxBias, s.genSnapshot.vizCtrnnMinAlpha, s.genSnapshot.vizCtrnnMaxAlpha, s.vizNonTunableHparams.Feedforward, 111, 11)
	renderHyperparams(s.screen, &s.currentHparams, s.pendingHparams, &s.requestedHparams, &s.vizNonTunableHparams, s.selectedHparamIdx, hparamsMetadata, 133, 0)
	renderControlsLegend(s.screen, s.isFrozenToggled, 133, 17)
	if s.isSugiyamaToggled {
		s.renderSugiyamaSection()
	} else {
		s.renderChampionsSection()
	}
	s.clearArea(width-21, width, 0, height)
	renderMaxFitnessHistory(s.screen, s.maxFitnessHistory, height-2, width-21, 0)
	s.screen.Show()
}

func (s *Server) renderChampionsSection() {
	width, height := s.screen.Size()
	s.clearArea(0, width-23, 18, height)
	renderChampionsHeader(s.screen, 0, 18)
	renderInputNames(s.screen, s.inputNames, 0, 22)
	renderPerformanceInfo(s.screen, s.genSnapshot.intervalSincePrev, s.perfRelatedParams, 0, height-7)
	maxInputNameLen := 0
	for _, name := range s.inputNames {
		if len(name) > maxInputNameLen {
			maxInputNameLen = len(name)
		}
	}
	for i, speciesId := range s.genSnapshot.speciesIdsSortedByChampions {
		champion := s.genSnapshot.topNetworksBySpeciesId[speciesId][0]
		xOffset := maxInputNameLen + 2 + (width/5)*i
		renderNetworkHeader(s.screen, champion.SpeciesId, champion.Fitness, false, false, xOffset, 19)
		renderNetworkInCompactLayout(s.screen, champion, s.genSnapshot.vizMinBias, s.genSnapshot.vizMaxBias, s.genSnapshot.vizCtrnnMinAlpha, s.genSnapshot.vizCtrnnMaxAlpha, s.genSnapshot.vizMinWeight, s.genSnapshot.vizMaxWeight, s.vizNonTunableHparams.Feedforward, xOffset, 20)
	}
}

func (s *Server) renderSugiyamaSection() {
	width, height := s.screen.Size()
	s.clearArea(0, width-23, 18, height)
	displayNetwork := s.genSnapshot.topNetworksBySpeciesId[s.genSnapshot.speciesIdsSortedByChampions[s.selectedSpeciesIdx]]
	maxInputNameLen := 0
	for _, name := range s.inputNames {
		if len(name) > maxInputNameLen {
			maxInputNameLen = len(name)
		}
	}
	baseXOffset := maxInputNameLen + 2
	renderSugiyamaViewHeader(s.screen, displayNetwork[s.selectedNetworkIdx].SpeciesId, displayNetwork[s.selectedNetworkIdx].Fitness, s.selectedNetworkIdx, len(displayNetwork), 0, 17)
	renderNetworkInSugiyamaLayout(s.screen, displayNetwork[s.selectedNetworkIdx], width-baseXOffset-23, height-20, baseXOffset, 20, s.genSnapshot.vizMinWeight, s.genSnapshot.vizMaxWeight, s.genSnapshot.vizMinBias, s.genSnapshot.vizMaxBias, s.genSnapshot.vizCtrnnMinAlpha, s.genSnapshot.vizCtrnnMaxAlpha, s.vizNonTunableHparams.Feedforward, s.inputNames, s.outputNames)
	for i, speciesId := range s.genSnapshot.speciesIdsSortedByChampions {
		champion := s.genSnapshot.topNetworksBySpeciesId[speciesId][0]
		xOffset := baseXOffset + (width/5)*i
		renderNetworkHeader(s.screen, champion.SpeciesId, champion.Fitness, s.isSugiyamaToggled, s.selectedSpeciesIdx == i, xOffset, 18)
	}
}

func parseVizTunableHparams(h *pb.VizTunableHparams) VizTunableHparams {
	return VizTunableHparams{
		AddOneConnProb:                    h.AddOneConnProb,
		AddOneNodeProb:                    h.AddOneNodeProb,
		EraseConnProb:                     h.EraseConnProb,
		PerturbWghtStdev:                  h.PerturbWghtStdev,
		ReplaceWghtStdev:                  h.ReplaceWghtStdev,
		PerturbWghtProb:                   h.PerturbWghtProb,
		ReplaceWghtProb:                   h.ReplaceWghtProb,
		PerturbBiasStdev:                  h.PerturbBiasStdev,
		ReplaceBiasStdev:                  h.ReplaceBiasStdev,
		PerturbBiasProb:                   h.PerturbBiasProb,
		ReplaceBiasProb:                   h.ReplaceBiasProb,
		PerturbAlphaStdev:                 h.PerturbAlphaStdev,
		PerturbAlphaProb:                  h.PerturbAlphaProb,
		ReplaceAlphaProb:                  h.ReplaceAlphaProb,
		ChangeActProb:                     h.ChangeActProb,
		ChangeAggProb:                     h.ChangeAggProb,
		DefaultOutputActivation:           h.DefaultOutputActivation,
		DisableCProb:                      h.DisableCProb,
		ReenableOneCProb:                  h.ReenableOneCProb,
		CrossoverRate:                     h.CrossoverRate,
		EnabledRecessivenessProb:          h.EnabledRecessivenessProb,
		AvgWeightsProb:                    h.AvgWeightsProb,
		IntraspeciesParenthoodEligibility: h.IntraspeciesParenthoodEligibility,
		SpeciesFitnessPowerscaling:        h.SpeciesFitnessPowerscaling,
		InterspeciesMatingRatio:           h.InterspeciesMatingRatio,
		CompatExcessCoeff:                 h.CompatExcessCoeff,
		CompatDisjointCoeff:               h.CompatDisjointCoeff,
		CompatWghtCoeff:                   h.CompatWghtCoeff,
		CompatEnabledCoeff:                h.CompatEnabledCoeff,
		CompatAggCoeff:                    h.CompatAggCoeff,
		CompatActCoeff:                    h.CompatActCoeff,
		CompatBiasCoeff:                   h.CompatBiasCoeff,
		CompatAlphaCoeff:                  h.CompatAlphaCoeff,
		BaseCompatThreshold:               h.BaseCompatThreshold,
		DynamicThresholds:                 h.DynamicThresholds,
		DynamicThresholdRatio:             h.DynamicThresholdRatio,
		DynamicThresholdLearningRate:      h.DynamicThresholdLearningRate,
		BaseStagnationAge:                 float32(h.BaseStagnationAge),
		StagnationAgePowerlawGrowth:       h.StagnationAgePowerlawGrowth,
		NStagnationExemptTopSpecies:       float32(h.NStagnationExemptTopSpecies),
	}
}

func serializeVizTunableHparams(v VizTunableHparams) *pb.VizTunableHparams {
	return &pb.VizTunableHparams{
		AddOneConnProb:                    v.AddOneConnProb,
		AddOneNodeProb:                    v.AddOneNodeProb,
		EraseConnProb:                     v.EraseConnProb,
		PerturbWghtStdev:                  v.PerturbWghtStdev,
		ReplaceWghtStdev:                  v.ReplaceWghtStdev,
		PerturbWghtProb:                   v.PerturbWghtProb,
		ReplaceWghtProb:                   v.ReplaceWghtProb,
		PerturbBiasStdev:                  v.PerturbBiasStdev,
		ReplaceBiasStdev:                  v.ReplaceBiasStdev,
		PerturbBiasProb:                   v.PerturbBiasProb,
		ReplaceBiasProb:                   v.ReplaceBiasProb,
		PerturbAlphaStdev:                 v.PerturbAlphaStdev,
		PerturbAlphaProb:                  v.PerturbAlphaProb,
		ReplaceAlphaProb:                  v.ReplaceAlphaProb,
		ChangeActProb:                     v.ChangeActProb,
		ChangeAggProb:                     v.ChangeAggProb,
		DefaultOutputActivation:           v.DefaultOutputActivation,
		DisableCProb:                      v.DisableCProb,
		ReenableOneCProb:                  v.ReenableOneCProb,
		CrossoverRate:                     v.CrossoverRate,
		EnabledRecessivenessProb:          v.EnabledRecessivenessProb,
		AvgWeightsProb:                    v.AvgWeightsProb,
		IntraspeciesParenthoodEligibility: v.IntraspeciesParenthoodEligibility,
		SpeciesFitnessPowerscaling:        v.SpeciesFitnessPowerscaling,
		InterspeciesMatingRatio:           v.InterspeciesMatingRatio,
		CompatExcessCoeff:                 v.CompatExcessCoeff,
		CompatDisjointCoeff:               v.CompatDisjointCoeff,
		CompatWghtCoeff:                   v.CompatWghtCoeff,
		CompatEnabledCoeff:                v.CompatEnabledCoeff,
		CompatAggCoeff:                    v.CompatAggCoeff,
		CompatActCoeff:                    v.CompatActCoeff,
		CompatBiasCoeff:                   v.CompatBiasCoeff,
		CompatAlphaCoeff:                  v.CompatAlphaCoeff,
		BaseCompatThreshold:               v.BaseCompatThreshold,
		DynamicThresholds:                 v.DynamicThresholds,
		DynamicThresholdRatio:             v.DynamicThresholdRatio,
		DynamicThresholdLearningRate:      v.DynamicThresholdLearningRate,
		BaseStagnationAge:                 int32(v.BaseStagnationAge),
		StagnationAgePowerlawGrowth:       v.StagnationAgePowerlawGrowth,
		NStagnationExemptTopSpecies:       int32(v.NStagnationExemptTopSpecies),
	}
}

func parseVizNonTunableHparams(h *pb.VizHparams) VizNonTunableHparams {
	return VizNonTunableHparams{
		PopSize:                           h.PopSize,
		Feedforward:                       h.Feedforward,
		CtrnnIntegrationSteps:             h.CtrnnIntegrationSteps,
		IntraspeciesUnchangedFrontrunners: h.IntraspeciesUnchangedFrontrunners,
		NormalizeObs:                      h.NormalizeObs,
	}
}

func parseNetworks(networks []*pb.Network) []Network {
	result := make([]Network, len(networks))
	for i, network := range networks {
		result[i] = Network{
			Cs:          parseConnections(network.Cs),
			ToposrtdNds: parseNodes(network.ToposrtdNds),
			Fitness:     float32(network.Fitness),
			SpeciesId:   int(network.SpeciesId),
		}
	}
	return result
}
func parseConnections(conns []*pb.Connection) []C {
	result := make([]C, len(conns))
	for i, c := range conns {
		result[i] = C{
			InId:    int(c.InId),
			OutId:   int(c.OutId),
			Enabled: c.Enabled,
			Weight:  float32(c.Weight),
		}
	}
	return result
}
func parseNodes(nds []*pb.Node) []Nd {
	result := make([]Nd, len(nds))
	for i, nd := range nds {
		result[i] = Nd{
			Id:          int(nd.Id),
			Type:        int(nd.Type),
			Aggregation: int(nd.Aggregation),
			Activation:  int(nd.Activation),
			Bias:        float32(nd.Bias),
			CtrnnAlpha:  float32(nd.CtrnnAlpha),
		}
	}
	return result
}
func parseSpecies(species []*pb.Species) []Species {
	result := make([]Species, len(species))
	for i, sp := range species {
		result[i] = Species{
			Id:              int(sp.Id),
			Quota:           int(sp.Quota),
			MemberCount:     int(sp.MemberCount),
			MinFitness:      float32(sp.MinFitness),
			MaxFitness:      float32(sp.MaxFitness),
			CompatThreshold: float32(sp.CompatThreshold),
			Stagnation:      int(sp.Stagnation),
		}
	}
	return result
}

func parsePerformanceRelatedParams(ps *pb.PerformanceRelatedParams) PerformanceRelatedParams {
	return PerformanceRelatedParams{
		GpuEnabled:      ps.GpuEnabled,
		PopSize:         int(ps.PopSize),
		MaxRolloutSteps: int(ps.MaxRolloutSteps),
		RolloutRepeats:  int(ps.RolloutRepeats),
		MaxNodes:        int(ps.MaxNodes),
		MaxConns:        int(ps.MaxConns),
	}
}

func (s *Server) updateGenTimestamps() {
	now := time.Now()
	s.genTimestamps = append(s.genTimestamps, now)
	if len(s.genTimestamps) > 2 {
		s.genTimestamps = s.genTimestamps[len(s.genTimestamps)-2:]
	}
}

func (s *Server) updateSpeciesHistory(species []Species) {
	s.speciesHistory = append(s.speciesHistory, species)
	if len(s.speciesHistory) > 50 {
		s.speciesHistory = s.speciesHistory[len(s.speciesHistory)-50:]
	}
}

func (s *Server) updateMaxFitnessHistory(species []Species) {
	maxFitness := species[0].MaxFitness
	for _, sp := range species {
		if sp.MaxFitness > maxFitness {
			maxFitness = sp.MaxFitness
		}
	}
	s.maxFitnessHistory = append(s.maxFitnessHistory, maxFitness)
	if len(s.maxFitnessHistory) > 200 {
		s.maxFitnessHistory = s.maxFitnessHistory[len(s.maxFitnessHistory)-200:]
	}
}

func (s *Server) updateSnapshot(networks []Network, species []Species, speciesHistory [][]Species, vizMinWeight, vizMaxWeight, vizMinBias, vizMaxBias, vizCtrnnMinAlpha, vizCtrnnMaxAlpha float32, popAvgFitness float32, genTimestamps []time.Time) {
	s.genSnapshot.vizMinWeight = vizMinWeight
	s.genSnapshot.vizMaxWeight = vizMaxWeight
	s.genSnapshot.vizMinBias = vizMinBias
	s.genSnapshot.vizMaxBias = vizMaxBias
	s.genSnapshot.vizCtrnnMinAlpha = vizCtrnnMinAlpha
	s.genSnapshot.vizCtrnnMaxAlpha = vizCtrnnMaxAlpha
	s.genSnapshot.popAvgFitness = popAvgFitness
	s.genSnapshot.species = species
	if len(genTimestamps) >= 2 {
		s.genSnapshot.prevGenSpecies = speciesHistory[len(speciesHistory)-2]
		s.genSnapshot.intervalSincePrev = genTimestamps[len(genTimestamps)-1].Sub(s.genTimestamps[len(genTimestamps)-2]).Seconds()
	}
	s.genSnapshot.topNetworksBySpeciesId = make(map[int][]Network)
	for _, network := range networks {
		s.genSnapshot.topNetworksBySpeciesId[network.SpeciesId] = append(s.genSnapshot.topNetworksBySpeciesId[network.SpeciesId], network)
	}
	for speciesId := range s.genSnapshot.topNetworksBySpeciesId {
		sort.Slice(s.genSnapshot.topNetworksBySpeciesId[speciesId], func(i, j int) bool {
			return s.genSnapshot.topNetworksBySpeciesId[speciesId][i].Fitness > s.genSnapshot.topNetworksBySpeciesId[speciesId][j].Fitness
		})
	}
	s.genSnapshot.speciesIdsSortedByChampions = make([]int, 0, len(s.genSnapshot.topNetworksBySpeciesId))
	for speciesId := range s.genSnapshot.topNetworksBySpeciesId {
		s.genSnapshot.speciesIdsSortedByChampions = append(s.genSnapshot.speciesIdsSortedByChampions, speciesId)
	}
	sort.Slice(s.genSnapshot.speciesIdsSortedByChampions, func(i, j int) bool {
		return s.genSnapshot.topNetworksBySpeciesId[s.genSnapshot.speciesIdsSortedByChampions[i]][0].Fitness > s.genSnapshot.topNetworksBySpeciesId[s.genSnapshot.speciesIdsSortedByChampions[j]][0].Fitness
	})
}

func renderHeader(screen tcell.Screen, gen int, popAvgFitness float32, taskName string, x0 int, y0 int) {
	header1 := fmt.Sprintf("Task: %s", taskName)
	for i, r := range header1 {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	header2 := fmt.Sprintf("Generation: %2d   Avg. fit.: %.2f", gen, popAvgFitness)
	for i, r := range header2 {
		screen.SetContent(x0+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
}

func renderSpecies(screen tcell.Screen, species []Species, prevGenSpecies []Species, modifiedStagnationAge int32, x0 int, y0 int) {
	fitSpreadWidth := 15
	maxQuotaBarWidth := 15
	maxMemberBarWidth := 15
	speciesHeader := "Species:"
	for i, r := range speciesHeader {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	columnsHeader := " Id | Thr | Stg | Prev. f. spread  | Offspring quota | Assgned members"
	for i, r := range columnsHeader {
		screen.SetContent(x0+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	columnHeader2 := fmt.Sprintf("%d", modifiedStagnationAge)
	for i, r := range columnHeader2 {
		screen.SetContent(x0+12+i, y0+2, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	maxQuota := 0
	maxMemberCount := 0
	for _, s := range species {
		if s.Quota > maxQuota {
			maxQuota = s.Quota
		}
		if s.MemberCount > maxMemberCount {
			maxMemberCount = s.MemberCount
		}
	}
	for i, sp := range species {
		if sp.Quota > 0 || sp.MemberCount > 0 {
			id := fmt.Sprintf("%2d", sp.Id)
			for j, r := range id {
				screen.SetContent(x0+1+j, y0+3+i, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
			}
			thr := fmt.Sprintf("%4.1f", sp.CompatThreshold)
			for j, r := range thr {
				screen.SetContent(x0+5+j, y0+3+i, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
			}
			stg := fmt.Sprintf("%3d", sp.Stagnation)
			for j, r := range stg {
				screen.SetContent(x0+12+j, y0+3+i, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
			}
			if maxQuota > 0 {
				barWidth := int(float32(sp.Quota) * float32(maxQuotaBarWidth) / float32(maxQuota))
				if sp.Quota > 0 && barWidth == 0 {
					barWidth = 1
				}
				speciesColor := speciesColorGradient[sp.Id%len(speciesColorGradient)]
				for j := range barWidth {
					screen.SetContent(x0+22+fitSpreadWidth+j, y0+3+i, '█', nil, tcell.StyleDefault.Foreground(speciesColor))
				}
			}
			memberBarWidth := int(float32(sp.MemberCount) * float32(maxMemberBarWidth) / float32(maxMemberCount))
			if sp.MemberCount > 0 && memberBarWidth == 0 {
				memberBarWidth = 1
			}
			speciesColor := speciesColorGradient[sp.Id%len(speciesColorGradient)]
			for j := range memberBarWidth {
				screen.SetContent(x0+25+fitSpreadWidth+maxQuotaBarWidth+j, y0+3+i, '█', nil, tcell.StyleDefault.Foreground(speciesColor))
			}
		}
	}
	if len(prevGenSpecies) > 0 {
		prevGenMap := make(map[int]Species)
		for _, sp := range prevGenSpecies {
			prevGenMap[sp.Id] = sp
		}
		renderFitnessSpread(screen, species, prevGenSpecies, prevGenMap, x0+18, y0, fitSpreadWidth)
	}
}

func renderFitnessSpread(screen tcell.Screen, species []Species, sourceSpecies []Species, speciesMap map[int]Species, barX0 int, y0 int, barWidth int) {
	var minFitness, maxFitness float32
	if len(sourceSpecies) > 0 {
		minFitness = sourceSpecies[0].MinFitness
		maxFitness = sourceSpecies[0].MaxFitness

	}
	for _, sp := range sourceSpecies {
		if sp.MinFitness < minFitness {
			minFitness = sp.MinFitness
		}
		if sp.MaxFitness > maxFitness {
			maxFitness = sp.MaxFitness
		}
	}
	fitnessRange := maxFitness - minFitness
	if fitnessRange > 0 {
		minLabel := fmt.Sprintf("%.2f ", minFitness)
		for i, r := range minLabel {
			screen.SetContent(barX0+i, y0+2, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
		maxLabel := fmt.Sprintf(" %.2f", maxFitness)
		for i, r := range maxLabel {
			screen.SetContent(barX0+barWidth-len(maxLabel)+i, y0+2, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	} else {
		label := "One value only"
		for i, r := range label {
			screen.SetContent(barX0+i, y0+2, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	}

	for i, sp := range species {
		if sp.Quota > 0 {
			var targetSp Species
			if speciesMap != nil {
				if prevSp, existed := speciesMap[sp.Id]; existed {
					targetSp = prevSp
				} else {
					continue
				}
			} else {
				targetSp = sp
			}
			if fitnessRange > 0 {
				minPos := barX0 + int(float32(barWidth-1)*(targetSp.MinFitness-minFitness)/fitnessRange)
				maxPos := barX0 + int(float32(barWidth-1)*(targetSp.MaxFitness-minFitness)/fitnessRange)
				speciesColor := speciesColorGradient[sp.Id%len(speciesColorGradient)]
				for j := minPos; j <= maxPos; j++ {
					screen.SetContent(j, y0+3+i, '━', nil, tcell.StyleDefault.Foreground(speciesColor))
				}
			}
		}
	}
}

func renderSpeciesHistory(screen tcell.Screen, speciesHistory [][]Species, popSize int, render_last_n int, x0 int, y0 int) {
	if len(speciesHistory) == 0 {
		return
	}
	title := fmt.Sprintf("Member count (last %d g.):", render_last_n)
	for i, r := range title {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	barWidth := 38
	startIdx := 0
	if len(speciesHistory) > render_last_n {
		startIdx = len(speciesHistory) - render_last_n
	}
	for genIdx, species := range speciesHistory[startIdx:] {
		y := y0 + 1 + genIdx
		currentX := x0
		cumulativeMemberCount := 0
		for _, sp := range species {
			if sp.MemberCount > 0 {
				speciesColor := speciesColorGradient[sp.Id%len(speciesColorGradient)]
				cumulativeMemberCount += sp.MemberCount
				targetEndX := x0 + int(float32(cumulativeMemberCount)*float32(barWidth)/float32(popSize))
				endX := targetEndX
				if endX <= currentX {
					endX = currentX + 1
				}
				for x := currentX; x < endX; x++ {
					screen.SetContent(x, y, '█', nil, tcell.StyleDefault.Foreground(speciesColor))
				}
				currentX = endX
			}
		}
	}
}

func renderMaxFitnessHistory(screen tcell.Screen, maxFitnessHistory []float32, render_last_n int, x0 int, y0 int) {
	title := fmt.Sprintf("Max. f. (last %d g.):", render_last_n)
	for i, r := range title {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	if len(maxFitnessHistory) == 0 {
		return
	}
	startIdx := 0
	if len(maxFitnessHistory) > render_last_n {
		startIdx = len(maxFitnessHistory) - render_last_n
	}
	historyToRender := maxFitnessHistory[startIdx:]
	minFitness := historyToRender[0]
	maxFitness := historyToRender[0]
	for _, fitness := range historyToRender {
		if fitness < minFitness {
			minFitness = fitness
		}
		if fitness > maxFitness {
			maxFitness = fitness
		}
	}
	barWidth := 21
	fitnessRange := maxFitness - minFitness
	if fitnessRange > 0 {
		minLabel := fmt.Sprintf("%.2f", minFitness)
		for i, r := range minLabel {
			screen.SetContent(x0+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
		maxLabel := fmt.Sprintf("%.2f", maxFitness)
		for i, r := range maxLabel {
			screen.SetContent(x0+barWidth-len(maxLabel)+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	} else {
		label := fmt.Sprintf("%.2f", minFitness)
		for i, r := range label {
			screen.SetContent(x0+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	}
	for i, fitness := range historyToRender {
		y := y0 + 2 + i
		var barLength int
		if fitnessRange > 0 {
			barLength = int(float32(barWidth) * (fitness - minFitness) / fitnessRange)
		} else {
			barLength = barWidth / 2
		}
		if barLength < 1 {
			barLength = 1
		}
		for x := 0; x < barLength; x++ {
			screen.SetContent(x0+x, y, '█', nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	}
}

func renderPerformanceInfo(screen tcell.Screen, interval float64, stats PerformanceRelatedParams, x0 int, y0 int) {
	var timingStr string
	if interval > 0 {
		timingStr = fmt.Sprintf("%.1fs", interval)
	} else {
		timingStr = "N/A"
	}
	for i, r := range fmt.Sprintf("Prev. g.: %s", timingStr) {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	if stats.GpuEnabled {
		for i, r := range "GPU: Yes" {
			screen.SetContent(x0+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	} else {
		for i, r := range "GPU: No" {
			screen.SetContent(x0+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	}
	for i, r := range fmt.Sprintf("Max rollout steps: %d", stats.MaxRolloutSteps) {
		screen.SetContent(x0+i, y0+2, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	for i, r := range fmt.Sprintf("Rollout repeats: %d", stats.RolloutRepeats) {
		screen.SetContent(x0+i, y0+3, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	for i, r := range fmt.Sprintf("Pop. size: %d", stats.PopSize) {
		screen.SetContent(x0+i, y0+4, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	for i, r := range fmt.Sprintf("Max nodes: %d", stats.MaxNodes) {
		screen.SetContent(x0+i, y0+5, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	for i, r := range fmt.Sprintf("Max conns: %d", stats.MaxConns) {
		screen.SetContent(x0+i, y0+6, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
}

func hparamEqual(field hparamMetadata, a, b *VizTunableHparams) bool {
	switch field.Kind {
	case hpFloat:
		return math.Abs(float64(*field.FloatPtr(a)-*field.FloatPtr(b))) <= 1e-4
	case hpBool:
		return *field.BoolPtr(a) == *field.BoolPtr(b)
	}
	return true
}

func hparamFormat(field hparamMetadata, h *VizTunableHparams) string {
	switch field.Kind {
	case hpFloat:
		return fmt.Sprintf("%.3f", *field.FloatPtr(h))
	case hpBool:
		if *field.BoolPtr(h) {
			return "Yes"
		}
		return "No"
	}
	return ""
}

func renderHyperparams(screen tcell.Screen, current *VizTunableHparams, pending *VizTunableHparams, requested *VizTunableHparams, nonTunable *VizNonTunableHparams, selectedIdx int, hparamsMetadata []hparamMetadata, x0 int, y0 int) {
	title := "Some hyperparams:"
	for i, r := range title {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	itemsPerCol := 14
	colWidth := 28
	renderIdx := 0
	for idx, field := range hparamsMetadata {
		if nonTunable.Feedforward {
			switch field.Description {
			case "Perturb alph stdev", "Perturb alpha p.", "Replace alpha p.":
				continue
			}
		}
		col := renderIdx / itemsPerCol
		row := renderIdx % itemsPerCol
		renderIdx++
		xOffset := x0 + col*colWidth
		yPos := y0 + 1 + row
		selector := "  "
		displaySrc := current
		style := tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255))
		if pending != nil && !hparamEqual(field, pending, current) {
			displaySrc = pending
			style = tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)).Background(pendingHparamColor)
		}
		if !hparamEqual(field, requested, current) {
			displaySrc = requested
			style = tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)).Background(requestedHparamColor)
			if pending != nil && !hparamEqual(field, pending, requested) {
				displaySrc = pending
				style = tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)).Background(pendingHparamColor)
			}
		}
		if idx == selectedIdx {
			selector = "> "
			style = style.Bold(true)
		}
		line := fmt.Sprintf("%s%-18s %s", selector, field.Description, hparamFormat(field, displaySrc))
		for i, r := range line {
			screen.SetContent(xOffset+i, yPos, r, nil, style)
		}
	}

	typeLabel := "CTRNN"
	if nonTunable.Feedforward {
		typeLabel = "Feedforward"
	}
	normalizeObsLabel := "No"
	if nonTunable.NormalizeObs {
		normalizeObsLabel = "Yes"
	}
	nonTunableLines := []string{
		fmt.Sprintf("  %-18s %d", "Pop. size", nonTunable.PopSize),
		fmt.Sprintf("  %-*s %s", 18+len(fmt.Sprintf("%d", nonTunable.PopSize))-len(typeLabel), "Type", typeLabel),
	}
	if !nonTunable.Feedforward {
		nonTunableLines = append(nonTunableLines, fmt.Sprintf("  %-18s %d", "CTRNN integ. steps", nonTunable.CtrnnIntegrationSteps))
	}
	nonTunableLines = append(nonTunableLines, fmt.Sprintf("  %-*s %s", 18+len(fmt.Sprintf("%d", nonTunable.PopSize))-len(normalizeObsLabel), "Normalize obs.", normalizeObsLabel))
	nonTunableLines = append(nonTunableLines, fmt.Sprintf("  %-18s %d", "Intraspcs unchangd", nonTunable.IntraspeciesUnchangedFrontrunners))
	lastIdx := renderIdx - 1
	ntCol := lastIdx / itemsPerCol
	ntRow := lastIdx%itemsPerCol + 2
	for _, line := range nonTunableLines {
		if ntRow >= itemsPerCol {
			ntRow = 0
			ntCol++
		}
		xOffset := x0 + ntCol*colWidth
		yPos := y0 + 1 + ntRow
		for i, r := range line {
			screen.SetContent(xOffset+i, yPos, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
		ntRow++
	}

	legendY := y0 + itemsPerCol + 1
	screen.SetContent(x0, legendY, ' ', nil, tcell.StyleDefault.Background(pendingHparamColor))
	for i, r := range " Pending" {
		screen.SetContent(x0+1+i, legendY, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	screen.SetContent(x0, legendY+1, ' ', nil, tcell.StyleDefault.Background(requestedHparamColor))
	for i, r := range " Requested, on the next g." {
		screen.SetContent(x0+1+i, legendY+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
}

func renderActivationsLegend(screen tcell.Screen, x0 int, y0 int) {
	title := "Activations:"
	for i, r := range title {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	keys := make([]int, 0, len(activationCodes))
	for k := range activationCodes {
		if k != -1 {
			keys = append(keys, k)
		}
	}
	sort.Ints(keys)
	for i, key := range keys {
		plotCode := activationCodes[key]
		screen.SetContent(x0, y0+i+1, '0', nil, tcell.StyleDefault.Foreground(plotCode.Color).Bold(true))
		label := fmt.Sprintf(" %s", plotCode.Description)
		for j, r := range label {
			screen.SetContent(x0+1+j, y0+i+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	}
}

func renderAggregationsLegend(screen tcell.Screen, x0 int, y0 int) {
	title := "Aggregations:"
	for i, r := range title {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	keys := make([]int, 0, len(aggregationCodes))
	for k := range aggregationCodes {
		if k != -1 {
			keys = append(keys, k)
		}
	}
	sort.Ints(keys)
	for i, key := range keys {
		plotCode := aggregationCodes[key]
		screen.SetContent(x0, y0+i+1, '0', nil, tcell.StyleDefault.Background(plotCode.Color).Foreground(activationCodes[1].Color).Bold(true))
		label := fmt.Sprintf(" %s", plotCode.Description)
		for j, r := range label {
			screen.SetContent(x0+1+j, y0+i+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
		}
	}
}

func renderGradientLegend(screen tcell.Screen, title string, minVal float32, maxVal float32, gradientWidth int, x0 int, y0 int) {
	for i, r := range title {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
	for i := range gradientWidth {
		position := float32(i) / float32(gradientWidth-1)
		value := minVal + (maxVal-minVal)*position
		if value > maxVal {
			value = maxVal
		} else if value < minVal {
			value = minVal
		}
		color, err := weightColorGradient.pick(value, minVal, maxVal)
		if err == nil {
			screen.SetContent(x0+i, y0+1, '█', nil, tcell.StyleDefault.Foreground(color))
		}
	}
	minLabel := fmt.Sprintf("%.2f", minVal)
	for i, r := range minLabel {
		position := float32(i) / float32(gradientWidth-1)
		value := minVal + (maxVal-minVal)*position
		if value > maxVal {
			value = maxVal
		} else if value < minVal {
			value = minVal
		}
		bgColor, _ := weightColorGradient.pick(value, minVal, maxVal)
		screen.SetContent(x0+i, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(50, 50, 50)).Background(bgColor).Bold(true))
	}
	maxLabel := fmt.Sprintf("%.2f", maxVal)
	for i, r := range maxLabel {
		pos := gradientWidth - len(maxLabel) + i
		position := float32(pos) / float32(gradientWidth-1)
		value := minVal + (maxVal-minVal)*position
		if value > maxVal {
			value = maxVal
		} else if value < minVal {
			value = minVal
		}
		bgColor, _ := weightColorGradient.pick(value, minVal, maxVal)
		screen.SetContent(x0+pos, y0+1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(50, 50, 50)).Background(bgColor).Bold(true))
	}
}

func renderGradientLegends(screen tcell.Screen, vizMinWeight, vizMaxWeight, vizMinBias, vizMaxBias, vizCtrnnMinAlpha, vizCtrnnMaxAlpha float32, feedforward bool, x0 int, y0 int) {
	renderGradientLegend(screen, "Conn weights (v):", vizMinWeight, vizMaxWeight, 20, x0, y0)
	renderGradientLegend(screen, "Node bias (&):", vizMinBias, vizMaxBias, 20, x0, y0+2)
	if !feedforward {
		renderGradientLegend(screen, "Ctrnn alpha (%):", vizCtrnnMinAlpha, vizCtrnnMaxAlpha, 20, x0, y0+4)
	}
}

func renderChampionsHeader(screen tcell.Screen, x0 int, y0 int) {
	header := "Champions:"
	for i, r := range header {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
}

func renderSugiyamaViewHeader(screen tcell.Screen, speciesId int, fitness float32, idxInSpeciesDisplayCohort int, amountInSpeciesDisplayCohort int, x0 int, y0 int) {
	header := fmt.Sprintf("Species-%d top %d networks; (%d/%d) Fitness %.2f", speciesId, amountInSpeciesDisplayCohort, idxInSpeciesDisplayCohort+1, amountInSpeciesDisplayCohort, fitness)
	for i, r := range header {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)))
	}
}

func renderNetworkHeader(screen tcell.Screen, speciesId int, fitness float32, isSugiyamaActive bool, isSelected bool, x0 int, y0 int) {
	speciesStr := fmt.Sprintf("%.2f     %2d", fitness, speciesId)
	var bgColor tcell.Color
	if isSugiyamaActive {
		if isSelected {
			bgColor = tcell.NewRGBColor(200, 200, 50)
		} else {
			bgColor = tcell.NewRGBColor(80, 80, 80)
		}
	} else {
		bgColor = speciesColorGradient[speciesId%len(speciesColorGradient)]
	}
	for i, r := range speciesStr {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(255, 255, 255)).Background(bgColor))
	}
}

func renderInputNames(screen tcell.Screen, inputNames []string, x0 int, y0 int) {
	for i, name := range inputNames {
		for j, r := range name {
			screen.SetContent(x0+j, y0+i, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(128, 128, 128)))
		}
	}
}

func renderControlsLegend(screen tcell.Screen, isFrozenToggled bool, x0 int, y0 int) {
	prefix := "Controls: ↑↓ ←→ | t toggle networks | "
	for i, r := range prefix {
		screen.SetContent(x0+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(128, 128, 128)))
	}
	freezeText := "f freeze viz"
	style := tcell.StyleDefault.Foreground(tcell.NewRGBColor(128, 128, 128))
	if isFrozenToggled {
		style = tcell.StyleDefault.Foreground(tcell.NewRGBColor(0, 0, 0)).Background(tcell.NewRGBColor(200, 200, 50))
	}
	for i, r := range freezeText {
		screen.SetContent(x0+len(prefix)+i, y0, r, nil, style)
	}
	suffix := " | z quit"
	for i, r := range suffix {
		screen.SetContent(x0+len(prefix)+len(freezeText)+i, y0, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(128, 128, 128)))
	}
}

func calculateNetworkNodeCoordinates(network Network, x0 int, y0 int) map[int]NdPlotData {
	var inputs, outputs, hidden []Nd
	for _, nd := range network.ToposrtdNds {
		switch nd.Type {
		case 1:
			inputs = append(inputs, nd)
		case 2:
			outputs = append(outputs, nd)
		case 3:
			hidden = append(hidden, nd)
		}
	}

	// Sort the inputs and outputs by id - Their ids don't change
	sort.Slice(inputs, func(i, j int) bool {
		return inputs[i].Id < inputs[j].Id
	})
	sort.Slice(outputs, func(i, j int) bool {
		return outputs[i].Id < outputs[j].Id
	})
	var toposrtdNds []Nd
	toposrtdNds = append(toposrtdNds, inputs...)
	toposrtdNds = append(toposrtdNds, hidden...)
	toposrtdNds = append(toposrtdNds, outputs...)

	inputsCount := len(inputs)
	hiddenCount := len(hidden)
	xCoordinates := make(map[int]int)
	yCoordinates := make(map[int]int)
	inputsCounter := 0
	hiddenCounter := 0
	outputsCounter := 0
	for _, nd := range toposrtdNds {
		switch nd.Type {
		case 1:
			xCoordinates[nd.Id] = 0
			yCoordinates[nd.Id] = inputsCounter
			inputsCounter++
		case 2:
			xCoordinates[nd.Id] = 1 + hiddenCount + outputsCounter
			yCoordinates[nd.Id] = inputsCount + hiddenCount
			outputsCounter++
		case 3:
			xCoordinates[nd.Id] = 1 + hiddenCounter
			yCoordinates[nd.Id] = inputsCount + hiddenCounter
			hiddenCounter++
		}
	}
	idToNdPlotData := map[int]NdPlotData{}
	for _, nd := range toposrtdNds {
		idToNdPlotData[nd.Id] = NdPlotData{
			X:  xCoordinates[nd.Id] + x0,
			Y:  yCoordinates[nd.Id] + y0,
			Nd: nd,
		}
	}
	return idToNdPlotData
}

func renderNetworkNodes(screen tcell.Screen, idToNdPlotData map[int]NdPlotData, vizMinBias float32, vizMaxBias float32, vizCtrnnMinAlpha float32, vizCtrnnMaxAlpha float32, feedforward bool, yOffset int) {
	for _, plotData := range idToNdPlotData {
		actColor := activationCodes[plotData.Nd.Activation].Color
		aggColor := aggregationCodes[plotData.Nd.Aggregation].Color
		screen.SetContent(plotData.X, plotData.Y, '0', nil, tcell.StyleDefault.Foreground(actColor).Background(aggColor).Bold(true))
		if plotData.Nd.Type != 1 {
			biasColor, err := weightColorGradient.pick(plotData.Nd.Bias, vizMinBias, vizMaxBias)
			if err != nil {
				fmt.Println("Error:", err)
			} else {
				screen.SetContent(plotData.X, yOffset+1, '&', nil, tcell.StyleDefault.Foreground(biasColor))
			}
			if !feedforward {
				alphaColor, err := weightColorGradient.pick(plotData.Nd.CtrnnAlpha, vizCtrnnMinAlpha, vizCtrnnMaxAlpha)
				if err != nil {
					fmt.Println("Error:", err)
				} else {
					screen.SetContent(plotData.X, yOffset, '%', nil, tcell.StyleDefault.Foreground(alphaColor))
				}
			}
		}
	}
}

func renderNetworkConnections(screen tcell.Screen, connections []C, idToNdPlotData map[int]NdPlotData, vizMinWeight float32, vizMaxWeight float32) {
	for _, c := range connections {
		x := idToNdPlotData[c.OutId].X
		y := idToNdPlotData[c.InId].Y
		if !c.Enabled {
			screen.SetContent(x, y, 'v', nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(128, 128, 128)))
		} else {
			color, err := weightColorGradient.pick(c.Weight, vizMinWeight, vizMaxWeight)
			if err != nil {
				fmt.Println("Error:", err)
			} else {
				screen.SetContent(x, y, 'v', nil, tcell.StyleDefault.Foreground(color))
			}
		}
	}
}

func renderNetworkInCompactLayout(screen tcell.Screen, network Network, vizMinBias float32, vizMaxBias float32, vizCtrnnMinAlpha float32, vizCtrnnMaxAlpha float32, vizMinWeight float32, vizMaxWeight float32, feedforward bool, x0 int, y0 int) {
	idToNdPlotData := calculateNetworkNodeCoordinates(network, x0, y0+2)
	renderNetworkNodes(screen, idToNdPlotData, vizMinBias, vizMaxBias, vizCtrnnMinAlpha, vizCtrnnMaxAlpha, feedforward, y0)
	renderNetworkConnections(screen, network.Cs, idToNdPlotData, vizMinWeight, vizMaxWeight)
}

func (s *Server) runEventLoop() {
	for ev := range s.screen.EventQ() {
		switch ev := ev.(type) {
		case *tcell.EventKey:
			s.handleKeyEvent(ev)
		case *tcell.EventResize:
			s.mu.Lock()
			s.render()
			s.mu.Unlock()
		}
	}
}

func (s *Server) handleKeyEvent(ev *tcell.EventKey) {
	s.mu.Lock()
	defer s.mu.Unlock()
	switch ev.Key() {
	case tcell.KeyUp:
		if !s.isSugiyamaToggled {
			s.selectedHparamIdx--
			if s.selectedHparamIdx < 0 {
				s.selectedHparamIdx = len(hparamsMetadata) - 1
			}
		} else {
			s.selectedNetworkIdx--
			if s.selectedNetworkIdx < 0 {
				s.selectedNetworkIdx++
			}
		}
	case tcell.KeyDown:
		if !s.isSugiyamaToggled {
			s.selectedHparamIdx++
			if s.selectedHparamIdx >= len(hparamsMetadata) {
				s.selectedHparamIdx = 0
			}
		} else {
			s.selectedNetworkIdx++
			if s.selectedNetworkIdx >= len(s.genSnapshot.topNetworksBySpeciesId[s.genSnapshot.speciesIdsSortedByChampions[s.selectedSpeciesIdx]]) {
				s.selectedNetworkIdx--
			}
		}
	case tcell.KeyLeft:
		if !s.isSugiyamaToggled {
			s.updatePendingHparam(-1)
		} else {
			s.selectedSpeciesIdx--
			if s.selectedSpeciesIdx < 0 {
				s.selectedSpeciesIdx = len(s.genSnapshot.topNetworksBySpeciesId) - 1
			}
			s.selectedNetworkIdx = 0
		}
	case tcell.KeyRight:
		if !s.isSugiyamaToggled {
			s.updatePendingHparam(1)
		} else {
			s.selectedSpeciesIdx++
			if s.selectedSpeciesIdx >= len(s.genSnapshot.topNetworksBySpeciesId) {
				s.selectedSpeciesIdx = 0
			}
			s.selectedNetworkIdx = 0
		}
	case tcell.KeyRune:
		switch ev.Str() {
		case "t":
			if len(s.genSnapshot.speciesIdsSortedByChampions) == 0 {
				s.renderUninitializedMessage()
				return
			}
			s.isSugiyamaToggled = !s.isSugiyamaToggled
			if !s.isSugiyamaToggled {
				s.renderChampionsSection()
			}
		case "f":
			if len(s.genSnapshot.speciesIdsSortedByChampions) == 0 {
				s.renderUninitializedMessage()
				return
			}
			s.isFrozenToggled = !s.isFrozenToggled
			renderControlsLegend(s.screen, s.isFrozenToggled, 133, 17)
		case "z":
			s.screen.Fini()
			return
		}
	}
	if !s.isSugiyamaToggled {
		renderHyperparams(s.screen, &s.currentHparams, s.pendingHparams, &s.requestedHparams, &s.vizNonTunableHparams, s.selectedHparamIdx, hparamsMetadata, 133, 0)
	} else {
		s.renderSugiyamaSection()
	}
	s.screen.Show()
}

func (s *Server) updatePendingHparam(direction int) {
	if s.pendingHparams == nil {
		pendingHparams := s.currentHparams
		s.pendingHparams = &pendingHparams
	}
	field := hparamsMetadata[s.selectedHparamIdx]
	switch field.Kind {
	case hpFloat:
		ptr := field.FloatPtr(s.pendingHparams)
		*ptr += float32(direction) * field.Step
	case hpBool:
		ptr := field.BoolPtr(s.pendingHparams)
		*ptr = !*ptr
	}
}

func (s *Server) renderUninitializedMessage() {
	width, height := s.screen.Size()
	line1 := fmt.Sprintf("Listening on localhost:%s", s.port)
	line2 := "Send training data (python -m examples.train_<example> --visualize-training)"
	for i, r := range line1 {
		s.screen.SetContent(width/2-len(line1)/2+i, height-2, r, nil, tcell.StyleDefault)
	}
	for i, r := range line2 {
		s.screen.SetContent(width/2-len(line2)/2+i, height-1, r, nil, tcell.StyleDefault)
	}
	s.screen.Show()
	time.Sleep(250 * time.Millisecond)
	for i, r := range line1 {
		s.screen.SetContent(width/2-len(line1)/2+i, height-2, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(0, 0, 0)).Background(tcell.NewRGBColor(200, 200, 50)))
	}
	for i, r := range line2 {
		s.screen.SetContent(width/2-len(line2)/2+i, height-1, r, nil, tcell.StyleDefault.Foreground(tcell.NewRGBColor(0, 0, 0)).Background(tcell.NewRGBColor(200, 200, 50)))
	}
	s.screen.Show()
	time.Sleep(250 * time.Millisecond)
	for i, r := range line1 {
		s.screen.SetContent(width/2-len(line1)/2+i, height-2, r, nil, tcell.StyleDefault)
	}
	for i, r := range line2 {
		s.screen.SetContent(width/2-len(line2)/2+i, height-1, r, nil, tcell.StyleDefault)
	}
	s.screen.Show()
}
